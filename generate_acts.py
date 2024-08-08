import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
from nnsight import LanguageModel

DEBUG = False
if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}

config = configparser.ConfigParser()
config.read('config.ini')

def load_model(model_name, device='remote'):
    print(f"Loading model {model_name}...")
    weights_directory = config[model_name]['weights_directory']
    if device == 'remote':
        model = LanguageModel(weights_directory)
    else:
        model = LanguageModel(weights_directory, torch_dtype=t.bfloat16, device_map="auto")
    return model

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_acts(statements, model, layers, remote=True):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    acts = {}
    with model.trace(statements, remote=remote, **tracer_kwargs):
        for layer in layers:
            acts[layer] = model.model.layers[layer].output[0][:,-1,:].save()

    for layer, act in acts.items():
        acts[layer] = act.value
    
    return acts

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="llama-13b",
                        help="Size of the model to use. Options are 7B or 30B")
    parser.add_argument("--layers", nargs='+', type=int,
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to")
    parser.add_argument("--noperiod", action="store_true", default=False,
                        help="Set flag if you don't want to add a period to the end of each statement")
    parser.add_argument("--device", default="remote")
    args = parser.parse_args()

    t.set_grad_enabled(False)
    model = load_model(args.model, args.device)
    for dataset in args.datasets:
        statements = load_statements(dataset)
        if args.noperiod:
            statements = [statement[:-1] for statement in statements]
        layers = args.layers
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        save_dir = os.path.join(f"{args.output_dir}", args.model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.noperiod:
            save_dir = os.path.join(save_dir, "noperiod")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in tqdm(range(0, len(statements), 25)):
            acts = get_acts(statements[idx:idx + 25], model, layers, args.device == 'remote')
            for layer, act in acts.items():
                    t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")