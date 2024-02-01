import torch as t
import pandas as pd
import os
from generate_acts import load_model
from tqdm import tqdm
import argparse
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_few_shot_accuracy(datasets, model, n_shots=5, batch_size=32, calibrated=True, remote=True):
    """Compute the few-shot accuracy of the model on the given datasets.
    Returns a list of dictionaries with experimental results, namely:
    * The dataset used.
    * The number of shots in the few shot prompt.
    * The few shot prompt used.
    * The accuracy of the model.
    * The calibration constant, if calibrated=True.
    """

    # change padding sight to right
    model.tokenizer.padding_side = 'right'

    outs = []
    for dataset in datasets:
        out = {
            'dataset' : dataset,
            'n_shots' : n_shots
            }

        # prepare data and prompt
        data_directory = os.path.join(ROOT, 'datasets', f"{dataset}.csv")
        df = pd.read_csv(data_directory)
        shots = df.sample(n_shots)
        queries = df.drop(shots.index)

        prompt = ''
        for _, shot in tqdm(shots.iterrows(), desc=f'Processing {dataset}'):
            prompt += f'{shot["statement"]} '
            if bool(shot['label']):
                prompt += 'TRUE\n'
            else:
                prompt += 'FALSE\n'

        out['shots'] = shots['statement'].tolist()
        out['prompt'] = prompt

        # cache activations over the prompt for reuse
        with model.forward(output_hidden_states=True, remote=remote, remote_include_output=remote) as runner:
            with runner.invoke(prompt):
                pass
        past_key_values = runner.output['past_key_values']

        # get completions and evaluate accuracy
        true_idx, false_idx = model.tokenizer.encode(' TRUE')[-1], model.tokenizer.encode(' FALSE')[-1]
        diffs = []
        for batch_idx in range(0, len(queries), batch_size):
            batch = queries.iloc[batch_idx:batch_idx+batch_size]['statement'].tolist()

            # # prepare past_key_values
            # pkv_batch = tuple((
            #     past_key_values[layer][0].expand(len(batch), *past_key_values[layer][0].shape[1:]),
            #     past_key_values[layer][1].expand(len(batch), *past_key_values[layer][1].shape[1:])
            # ) for layer in range(len(past_key_values))
            # )

            batch_lens = [len(model.tokenizer.encode(query, add_special_tokens=False)) for query in batch]
            with model.forward(past_key_values=past_key_values
            , remote=remote, remote_include_output=False) as runner:
                with runner.invoke(batch, add_special_tokens=False, return_attention_mask=False):
                    logits = model.lm_head.output
                    logits = logits[t.arange(len(batch)), t.tensor(batch_lens) - 1, :]
                    probs = logits.softmax(-1)
                    diffs.append((probs[:, true_idx] - probs[:, false_idx]).save())
        diffs = t.cat([diff.value for diff in diffs])


        # if calibrated, compute calibration constant
        if calibrated:
            gamma = t.sort(diffs).values[len(diffs) // 2]
            out['gamma'] = gamma.item()
        else:
            gamma = 0
        
        # get predicted labels
        predicted_labels = diffs > gamma
        ground_truth = t.tensor(queries['label'].values, device=predicted_labels.device).bool()
        
        acc = (predicted_labels == ground_truth).float().mean().item()
        out['acc'] = acc

        outs.append(out)

    return outs

if __name__ == '__main__':
    """
    Compute the few-shot accuracy of the model on the given datasets and save results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to evaluate on')
    parser.add_argument('--model', type=str, default='llama-2-70b', help='model size to evaluate')
    parser.add_argument('--n_shots', type=int, default=5, help='number of shots to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to use')
    parser.add_argument('--uncalibrated', action='store_true', default=False, help='set flag if using uncalibrated few shot')
    parser.add_argument('--device', default='remote', help='device to use')

    args = parser.parse_args()

    model = load_model(args.model, device=args.device)

    outs = get_few_shot_accuracy(args.datasets, model, args.n_shots, args.batch_size, not args.uncalibrated, args.device == 'remote')
    for out in outs:
        out['model'] = args.model

    # save results
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'r') as f:
        data = json.load(f)
    data.extend(outs)
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'w') as f:
        json.dump(data, f, indent=4)