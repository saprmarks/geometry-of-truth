import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
LLAMA_DIRECTORY = config['LLaMA']['weights_directory']

if not os.path.exists(LLAMA_DIRECTORY):
    raise Exception("Make sure you've set the path to your LLaMA weights in config.ini")

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs


def load_llama(model_size, device):
    llama_path = os.path.join(LLAMA_DIRECTORY, config['LLaMA'][f'{model_size}_subdir'])
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    model = LlamaForCausalLM.from_pretrained(llama_path)
    # set tokenizer to use bos token
    tokenizer.bos_token = '<s>'
    if model_size == '13B' and device != 'cpu':
        model = model.half()
    model.to(device)
    return tokenizer, model

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_acts(statements, tokenizer, model, layers, device):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    
    # get activations
    acts = {layer : [] for layer in layers}
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])
    
    for layer, act in acts.items():
        acts[layer] = t.stack(act).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="13B",
                        help="Size of the model to use. Options are 7B or 30B")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t.set_grad_enabled(False)
    
    tokenizer, model = load_llama(args.model, args.device)
    for dataset in args.datasets:
        statements = load_statements(dataset)
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        save_dir = f"{args.output_dir}/{args.model}/{dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(0, len(statements), 25):
            acts = get_acts(statements[idx:idx + 25], tokenizer, model, layers, args.device)
            for layer, act in acts.items():
                    t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")