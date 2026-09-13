import argparse
import json
import os

import torch as t

from generate_acts import (
    ROOT,
    TRACER_KWARGS,
    load_model,
    materialize,
    residual_stream,
)


def patching_experiment(model_name, continuation_idx=None, device="remote"):
    model = load_model(model_name, device=device)
    layers = model.model.layers
    remote = device == "remote"

    # prompt for sp_en_trans
    false_prompt = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
The Spanish word 'con' means 'one'. This statement is:"""
    true_prompt = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
The Spanish word 'uno' means 'one'. This statement is:"""

    false_toks = model.tokenizer(false_prompt).input_ids
    true_toks = model.tokenizer(true_prompt).input_ids
    if len(false_toks) != len(true_toks):
        raise ValueError(
            f"False prompt has length {len(false_toks)} but true prompt has length {len(true_toks)}"
        )

    sames = [false_tok == true_tok for false_tok, true_tok in zip(false_toks, true_toks)]
    n_toks = sames[::-1].index(False) + 1

    true_acts = []
    with model.trace(true_prompt, remote=remote, **TRACER_KWARGS):
        for layer in model.model.layers:
            true_acts.append(residual_stream(layer.output).save())
    true_acts = [materialize(act) for act in true_acts]

    results_path = os.path.join(ROOT, "experimental_outputs", "patching_results.json")
    if continuation_idx is not None:
        with open(results_path, "r") as f:
            outs = json.load(f)
        out = outs[continuation_idx]
        assert out["model"] == model_name
        assert out["false_prompt"] == false_prompt
        assert out["true_prompt"] == true_prompt
        logit_diffs = out["logit_diffs"]
    else:
        out = {
            "model": model_name,
            "false_prompt": false_prompt,
            "true_prompt": true_prompt,
        }
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
        out["logit_diffs"] = logit_diffs
        with open(results_path, "r") as f:
            outs = json.load(f)
        outs.append(out)
        with open(results_path, "w") as f:
            json.dump(outs, f, indent=4)
        continuation_idx = -1

    t_tok = model.tokenizer(" TRUE").input_ids[-1]
    f_tok = model.tokenizer(" FALSE").input_ids[-1]

    for tok_idx in range(1, n_toks + 1):
        for layer_idx, layer in enumerate(model.model.layers):
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue
            with model.trace(false_prompt, remote=remote, **TRACER_KWARGS):
                hidden = residual_stream(layer.output)
                hidden[0, -tok_idx, :] = true_acts[layer_idx][0, -tok_idx, :]
                logits = model.lm_head.output
                logit_diff = (logits[0, -1, t_tok] - logits[0, -1, f_tok]).save()
            logit_diffs[tok_idx - 1][layer_idx] = materialize(logit_diff).item()

            outs[continuation_idx] = out
            with open(results_path, "w") as f:
                json.dump(outs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-2-70b")
    parser.add_argument("--continuation_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default="remote")
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device)
