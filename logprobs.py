from nnsight import LanguageModel
import pandas as pd
import torch as t
import argparse
import os
from generate_acts import load_model


def compute_logprobs(model, dataset, remote=True):

    df = pd.read_csv(f'datasets/{dataset}.csv')

    all_logprobs = []
    # for each statement, get the logprob of the statement
    for statement in df['statement'].tolist():
        with model.forward(remote=remote, remote_include_output=remote) as runner:
            with runner.invoke(statement):
                logprobs = model.lm_head.output.log_softmax(dim=-1)
                tokens = runner.batched_input['input_ids'][0][1:]
                summed = logprobs[0, t.arange(len(tokens)), tokens].sum().save()
        all_logprobs.append(summed.value.item())
    
    df['logprob'] = all_logprobs

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute logprobs for statements in a dataset")
    parser.add_argument("--model", default="llama-2-70b")
    parser.add_argument("--dataset", default="cities")
    parser.add_argument("--device", default="remote")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

    remote = args.device == 'remote'

    df = compute_logprobs(model, args.dataset, remote=remote)

    df.to_csv(f'experimental_outputs/logprobs/{args.dataset}.csv')