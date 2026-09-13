import argparse
import json
import os

import pandas as pd
import torch as t
from probes import CCSProbe, LRProbe, MMProbe
from utils import collect_acts

from generate_acts import (
    ROOT,
    TRACER_KWARGS,
    config_flag,
    config_int,
    load_model,
    materialize,
    residual_stream,
    resolve_torch_device,
)

PROBES = {
    "LRProbe": LRProbe,
    "MMProbe": MMProbe,
    "CCSProbe": CCSProbe,
    "random": "random",
}


def intervention_experiment(
    model,
    queries,
    direction,
    hidden_states,
    intervention="none",
    batch_size=32,
    remote=True,
):
    """
    model : an nnsight LanguageModel
    queries : a list of statements to be labeled
    direction : a direction in the residual stream of the model
    hidden_states : list of (layer, -1 or 0) pairs, -1 for intervene before the period, 0 for intervene over the period
    intervention : 'none', 'add', or 'subtract'
    batch_size : batch size for forward passes
    remote : run on the NDIF server?
    Add the direction to the specified hidden states and return the resulting probability diff P(TRUE) - P(FALSE)
    and sum P(TRUE) + P(FALSE) averaged over the data
    """
    assert intervention in ["none", "add", "subtract"]
    if intervention == "add":
        delta = direction
    elif intervention == "subtract":
        delta = -direction
    else:
        delta = 0.0

    true_idx, false_idx = (
        model.tokenizer.encode(" TRUE")[-1],
        model.tokenizer.encode(" FALSE")[-1],
    )
    len_suffix = len(model.tokenizer.encode("This statement is:"))

    p_diffs = []
    tots = []
    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx : batch_idx + batch_size]
        with model.trace(batch, remote=remote, **TRACER_KWARGS):
            if intervention != "none":
                for layer, offset in hidden_states:
                    hidden = residual_stream(model.model.layers[layer].output)
                    hidden[:, -len_suffix + offset, :] += delta
            logits = model.lm_head.output[:, -1, :]
            probs = logits.softmax(-1)
            p_diffs.append((probs[:, true_idx] - probs[:, false_idx]).save())
            tots.append((probs[:, true_idx] + probs[:, false_idx]).save())
    p_diffs = t.cat([materialize(p_diff) for p_diff in p_diffs])
    tots = t.cat([materialize(tot) for tot in tots])

    return p_diffs.mean().item(), tots.mean().item()


def prepare_data(prompt, dataset, subset="all"):
    """
    prompt : the few shot prompt
    dataset : dataset name
    subset : 'all', 'true', or 'false'
    Returns a list of queries to be run through the model for the intervention experiment.
    """
    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{dataset}.csv"))
    if subset == "all":
        statements = df["statement"].tolist()
    elif subset == "true":
        statements = df[df["label"] == 1]["statement"].tolist()
    elif subset == "false":
        statements = df[df["label"] == 0]["statement"].tolist()
    else:
        raise ValueError(f"subset must be 'all', 'true', or 'false', not {subset}")

    queries = []
    for statement in statements:
        if statement not in prompt:
            queries.append(prompt + statement + " This statement is:")

    return queries


def _prompt_for(model_name, val_dataset):
    if model_name == "llama-2-70b" and val_dataset == "sp_en_trans":
        return """\
The Spanish word 'fruta' means 'goat'. This statement is: FALSE
The Spanish word 'carne' means 'meat'. This statement is: TRUE
"""
    if model_name == "llama-2-13b" and val_dataset == "sp_en_trans":
        return """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""
    raise ValueError(
        f"No hardcoded few-shot prompt for model={model_name} val_dataset={val_dataset}. "
        "Add one in interventions.py or use llama-2-13b/llama-2-70b with sp_en_trans."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-2-70b")
    parser.add_argument("--probe", default="LRProbe")
    parser.add_argument(
        "--train_datasets", nargs="+", default=["cities", "neg_cities"], type=str
    )
    parser.add_argument("--val_dataset", default="sp_en_trans", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--intervention", default="none", type=str)
    parser.add_argument("--subset", default="all", type=str)
    parser.add_argument("--device", default="remote", type=str)
    args = parser.parse_args()

    if args.probe not in PROBES:
        raise ValueError(f"Unknown probe {args.probe}. Options: {list(PROBES)}")
    ProbeClass = PROBES[args.probe]

    remote = args.device == "remote"
    probe_device = resolve_torch_device(args.device)

    model = load_model(args.model, args.device)

    start_layer = config_int(args.model, "intervene_layer")
    end_layer = config_int(args.model, "probe_layer")
    noperiod = config_flag(args.model, "noperiod")

    if noperiod:
        hidden_states = [(layer, -1) for layer in range(start_layer, end_layer + 1)]
    else:
        hidden_states = []
        for layer in range(start_layer, end_layer + 1):
            hidden_states.append((layer, -1))
            hidden_states.append((layer, 0))

    print("training probe...")
    if ProbeClass == LRProbe or ProbeClass == MMProbe or ProbeClass == "random":
        acts, labels = [], []
        for dataset in args.train_datasets:
            acts.append(
                collect_acts(
                    dataset, args.model, end_layer, noperiod=noperiod
                ).to(probe_device)
            )
            labels.append(
                t.Tensor(
                    pd.read_csv(os.path.join(ROOT, "datasets", f"{dataset}.csv"))[
                        "label"
                    ].tolist()
                ).to(probe_device)
            )
        acts, labels = t.cat(acts), t.cat(labels)
        if ProbeClass == LRProbe or ProbeClass == MMProbe:
            probe = ProbeClass.from_data(acts, labels, device=probe_device)
        else:
            probe = MMProbe.from_data(acts, labels, device=probe_device)
            probe.direction = t.nn.Parameter(t.randn_like(probe.direction))
    elif ProbeClass == CCSProbe:
        acts = collect_acts(
            args.train_datasets[0], args.model, end_layer, noperiod=noperiod
        ).to(probe_device)
        neg_acts = collect_acts(
            args.train_datasets[1], args.model, end_layer, noperiod=noperiod
        ).to(probe_device)
        labels = t.Tensor(
            pd.read_csv(
                os.path.join(ROOT, "datasets", f"{args.train_datasets[0]}.csv")
            )["label"].tolist()
        ).to(probe_device)
        probe = ProbeClass.from_data(
            acts, neg_acts, labels=labels, device=probe_device
        )

    direction = probe.direction
    true_acts, false_acts = acts[labels == 1], acts[labels == 0]
    true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
    direction = direction / direction.norm()
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    direction = direction.cpu()

    prompt = _prompt_for(args.model, args.val_dataset)
    queries = prepare_data(prompt, args.val_dataset, subset=args.subset)

    print("running intervention experiment...")
    p_diff, tot = intervention_experiment(
        model,
        queries,
        direction,
        hidden_states,
        intervention=args.intervention,
        batch_size=args.batch_size,
        remote=remote,
    )

    out = {
        "model": args.model,
        "train_datasets": args.train_datasets,
        "val_dataset": args.val_dataset,
        "probe class": ProbeClass if isinstance(ProbeClass, str) else ProbeClass.__name__,
        "prompt": prompt,
        "p_diff": p_diff,
        "tot": tot,
        "intervention": args.intervention,
        "subset": args.subset,
        "hidden_states": hidden_states,
    }

    results_path = os.path.join(
        ROOT, "experimental_outputs", "label_change_intervention_results.json"
    )
    with open(results_path, "r") as f:
        data = json.load(f)
    data.append(out)
    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)
