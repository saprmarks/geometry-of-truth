import argparse
import os

import pandas as pd
import torch as t

from generate_acts import ROOT, TRACER_KWARGS, load_model, materialize


def compute_logprobs(model, dataset, remote=True):
    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{dataset}.csv"))

    all_logprobs = []
    for statement in df["statement"].tolist():
        input_ids = model.tokenizer(statement, return_tensors="pt")["input_ids"][0]
        with model.trace(statement, remote=remote, **TRACER_KWARGS):
            logprobs = model.lm_head.output.log_softmax(dim=-1).save()
        logprobs = materialize(logprobs)
        n_pred = input_ids.shape[0] - 1
        summed = logprobs[0, t.arange(n_pred), input_ids[1:]].sum()
        all_logprobs.append(summed.item())

    df["logprob"] = all_logprobs
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute logprobs for statements in a dataset"
    )
    parser.add_argument("--model", default="llama-2-70b")
    parser.add_argument("--dataset", default="cities")
    parser.add_argument("--device", default="remote")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    remote = args.device == "remote"

    df = compute_logprobs(model, args.dataset, remote=remote)

    out_dir = os.path.join(ROOT, "experimental_outputs", "logprobs")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{args.dataset}.csv"))
