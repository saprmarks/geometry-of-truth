import torch as t
import pandas as pd
import os
from tqdm import tqdm
from utils import collect_acts
from generate_acts import load_model
from probes import LRProbe, MMProbe, CCSProbe
import plotly.express as px
import json
import argparse
import configparser

def intervention_experiment(model, queries, direction, hidden_states, intervention='none', batch_size=32, remote=True):
    """
    model : an nnsight LanguageModel
    queries : a list of statements to be labeled
    direction : a direction in the residual stream of the model
    hidden_states : list of (layer, -1 or 0) pairs, -1 for intervene before the period, 0 for intervene over the period
    subtract : if True, subtract the direction from the hidden states instead of adding it
    batch_size : batch size for forward passes
    remote : run on the NDIF server?
    Add the direction to the specified hidden states and return the resulting probability diff P(TRUE) - P(FALSE)
    and sum P(TRUE) + P(FALSE) averaged over the data
    """

    assert intervention in ['none', 'add', 'subtract']

    true_idx, false_idx = model.tokenizer.encode(' TRUE')[-1], model.tokenizer.encode(' FALSE')[-1]
    len_suffix = len(model.tokenizer.encode('This statement is:'))

    p_diffs = []
    tots = []
    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx+batch_size]
        with model.forward(remote=remote, remote_include_output=False) as runner:
            with runner.invoke(batch):
                for layer, offset in hidden_states:
                    model.model.layers[layer].output[0][:,-len_suffix + offset, :] += \
                        direction if intervention == 'add' else -direction if intervention == 'subtract' else 0.
                logits = model.lm_head.output[:, -1, :]
                probs = logits.softmax(-1)
                p_diffs.append((probs[:, true_idx] - probs[:, false_idx]).save())
                tots.append((probs[:, true_idx] + probs[:, false_idx]).save())
    p_diffs = t.cat([p_diff.value for p_diff in p_diffs])
    tots = t.cat([tot.value for tot in tots])

    return p_diffs.mean().item(), tots.mean().item()

def prepare_data(prompt, dataset, subset='all'):
    """
    prompt : the few shot prompt
    dataset : dataset name
    model : an nnsight LanguageModel
    subset : 'all', 'true', or 'false'
    Returns a list of queries to be run through the model for the patching experiment
    and a list of the index of the last period token in each query.
    """
    df = pd.read_csv(f'datasets/{dataset}.csv')
    if subset == 'all':
        statements = df['statement'].tolist()
    elif subset == 'true':
        statements = df[df['label'] == 1]['statement'].tolist()
    elif subset == 'false':
        statements = df[df['label'] == 0]['statement'].tolist()

    queries = []
    for statement in statements:
        if statement not in prompt:
            queries.append(prompt + statement + ' This statement is:')
    
    return queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama-2-70b')
    parser.add_argument('--probe', default='LRProbe')
    parser.add_argument('--train_datasets', nargs='+', default=['cities', 'neg_cities'], type=str)
    parser.add_argument('--val_dataset', default = 'sp_en_trans', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--intervention', default='none', type=str)
    parser.add_argument('--subset', default='all', type=str)
    parser.add_argument('--device', default='remote', type=str)
    args = parser.parse_args()

    remote = args.device == 'remote'

    model = load_model(args.model, args.device)

    # prepare hidden states to intervene over
    config = configparser.ConfigParser()
    config.read('config.ini')
    start_layer = eval(config[args.model]['intervene_layer'])
    end_layer = eval(config[args.model]['probe_layer'])
    noperiod = eval(config[args.model]['noperiod'])

    if noperiod:
        hidden_states = [
            (layer, -1) for layer in range(start_layer, end_layer + 1)
        ]
    else:
        hidden_states = []
        for layer in range(start_layer, end_layer + 1):
            hidden_states.append((layer, -1))
            hidden_states.append((layer, 0))
    
    print('training probe...')
    # get direction along which to intervene
    ProbeClass = eval(args.probe)
    if ProbeClass == LRProbe or ProbeClass == MMProbe or ProbeClass == 'random':
        acts, labels = [], []
        for dataset in args.train_datasets:
            acts.append(collect_acts(dataset, args.model, end_layer, noperiod=noperiod).to('cuda:0'))
            labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to('cuda:0'))
        acts, labels = t.cat(acts), t.cat(labels)
        if ProbeClass == LRProbe or ProbeClass == MMProbe:
            probe = ProbeClass.from_data(acts, labels, device='cuda:0')
        elif ProbeClass == 'random':
            probe = MMProbe.from_data(acts, labels, device='cuda:0')
            probe.direction = t.nn.Parameter(t.randn_like(probe.direction))
    elif ProbeClass == CCSProbe:
        acts = collect_acts(args.train_datasets[0], args.model, end_layer, noperiod=noperiod).to('cuda:0')
        neg_acts = collect_acts(args.train_datasets[1], args.model, end_layer, noperiod=noperiod).to('cuda:0')
        labels = t.Tensor(pd.read_csv(f'datasets/{args.train_datasets[0]}.csv')['label'].tolist()).to('cuda:0')
        probe = ProbeClass.from_data(acts, neg_acts, labels=labels, device='cuda:0')

    direction = probe.direction
    true_acts, false_acts = acts[labels==1], acts[labels==0]
    true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
    direction = direction / direction.norm()
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    direction = direction.cpu()

    # set prompt (hardcoded for now)
    if args.model == 'llama-2-70b' and args.val_dataset == 'sp_en_trans':
        prompt = """\
The Spanish word 'fruta' means 'goat'. This statement is: FALSE
The Spanish word 'carne' means 'meat'. This statement is: TRUE
"""
    elif args.model == 'llama-2-13b' and args.val_dataset == 'sp_en_trans':
        prompt = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""
    
    # prepare data
    queries = prepare_data(prompt, args.val_dataset, subset=args.subset)

    print('running intervention experiment...')
    # do intervention experiment
    p_diff, tot = intervention_experiment(model, queries, direction, hidden_states,
                                          intervention=args.intervention, batch_size=args.batch_size, remote=remote)

    # save results
    out = {
        'model' : args.model,
        'train_datasets' : args.train_datasets,
        'val_dataset' : args.val_dataset,
        'probe class' : ProbeClass.__name__,
        'prompt' : prompt,
        'p_diff' : p_diff,
        'tot' : tot,
        'intervention' : args.intervention,
        'subset' : args.subset,
        'hidden_states' : hidden_states,
    }

    with open('experimental_outputs/label_change_intervention_results.json', 'r') as f:
        data = json.load(f)
    data.append(out)
    with open('experimental_outputs/label_change_intervention_results.json', 'w') as f:
        json.dump(data, f, indent=4)
