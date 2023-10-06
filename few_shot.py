import torch as t
import pandas as pd
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import argparse
import json

t.set_grad_enabled(False)

LLAMA_DIRECTORY = '/home/ubuntu/llama_hf/'
ROOT = '/home/ubuntu/statement_reps/' # change to the location of this folder (geometry_of_truth)


def get_few_shot_accuracy(datasets, model_size, n_shots=5, calibrated=True, device='cpu'):
    """Compute the few-shot accuracy of the model on the given datasets.
    Returns a list of dictionaries with experimental results, namely:
    * The dataset used.
    * The number of shots in the few shot prompt.
    * The few shot prompt used.
    * The accuracy of the model.
    * The calibration constant, if calibrated=True.
    """

    # load LLaMA model 
    tokenizer = LlamaTokenizer.from_pretrained(os.path.join(LLAMA_DIRECTORY, model_size))
    model = LlamaForCausalLM.from_pretrained(os.path.join(LLAMA_DIRECTORY, 'llama_hf', model_size)).half().to(device)

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
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model(input_ids, output_hidden_states=True)
        past_key_values = outputs.past_key_values

        # get completions and evaluate accuracy
        true_idx, false_idx = tokenizer.encode('TRUE')[1], tokenizer.encode('FALSE')[1]
        p_trues, p_falses = [], []
        for _, query in tqdm(queries.iterrows()):
            input_ids = tokenizer.encode(prompt + query['statement'], return_tensors='pt').to(device)
            outputs = model(input_ids, past_key_values=past_key_values)
            probs = outputs.logits[0, -1, :].softmax(-1)
            p_trues.append(probs[true_idx].item()), p_falses.append(probs[false_idx].item())

        # if calibrated, compute calibration constant
        if calibrated:
            diffs = [p_true - p_false for p_true, p_false in zip(p_trues, p_falses)]
            gamma = sorted(diffs)[len(diffs) // 2]
            out['gamma'] = gamma
        else:
            gamma = 0
        
        # get predicted labels
        predicted_labels = [p_true - p_false > gamma for p_true, p_false in zip(p_trues, p_falses)]
        predicted_labels = t.Tensor(predicted_labels).to(device).float()
        true_labels = t.Tensor(queries['label'].values).to(device)
        
        acc = (predicted_labels == true_labels).float().mean().item()
        out['acc'] = acc

        outs.append(out)

    return outs

if __name__ == '__main__':
    """
    Compute the few-shot accuracy of the model on the given datasets and save results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to evaluate on')
    parser.add_argument('--model_size', type=str, default='13B', help='model size to evaluate')
    parser.add_argument('--n_shots', type=int, default=5, help='number of shots to use')
    parser.add_argument('--uncalibrated', action='store_true', default=False, help='set flag if using uncalibrated few shot')
    parser.add_argument('--device', default='cuda:0', help='device to use')

    args = parser.parse_args()

    out = get_few_shot_accuracy(args.datasets, args.model_size, args.n_shots, not args.uncalibrated, args.device)

    # save results
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'r') as f:
        data = json.load(f)
    data.extend(out)
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'w') as f:
        json.dump(data, f, indent=4)