import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
HF_KEY = config['hf_key']['hf_key']

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs

def load_model(model_name):
    print(f"Loading model {model_name}...")
    try:
        weights_directory = config[model_name]['weights_directory']
        TokenizerClass = eval(config[model_name]['tokenizer_class'])
        ModelClass = eval(config[model_name]['model_class'])
        tokenizer = TokenizerClass.from_pretrained(weights_directory, token=HF_KEY, torch_dtype=t.bfloat16, device_map="auto")
        model = ModelClass.from_pretrained(weights_directory, token=HF_KEY)
        all_layers = eval("model." + config[model_name]['layers'])
    except:
        raise ValueError("Cannot load model, make sure weights and huggingface key are set in config file")
    if model_name == 'llama-2-13b-reset':
        # create reset network by permuting the weights for each parameter
        for param in model.parameters():
            param.data = param.data[..., t.randperm(param.size(-1))]
    return tokenizer, model, all_layers 

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_acts(statements, tokenizer, model, layers, all_layers, device):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = all_layers[layer].register_forward_hook(hook)
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
    parser.add_argument("--model", default="llama-13b",
                        help="Size of the model to use. Options are 7B or 30B")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to")
    parser.add_argument("--noperiod", action="store_true", default=False,
                        help="Set flag if you don't want to add a period to the end of each statement")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t.set_grad_enabled(False)
    tokenizer, model, all_layers = load_model(args.model, args.device)
    for dataset in args.datasets:
        statements = load_statements(dataset)
        if args.noperiod:
            statements = [statement[:-1] for statement in statements]
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(all_layers)))
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

        for idx in range(0, len(statements), 25):
            acts = get_acts(statements[idx:idx + 25], tokenizer, model, layers, all_layers, args.device)
            for layer, act in acts.items():
                    t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")