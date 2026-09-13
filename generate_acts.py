import argparse
import configparser
import os

import pandas as pd
import torch as t
from nnsight import LanguageModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACER_KWARGS = {"scan": False, "validate": False}

config = configparser.ConfigParser()
config.read(os.path.join(ROOT, "config.ini"))


def config_int(model_name, key):
    return config.getint(model_name, key)


def config_flag(model_name, key):
    return config.getboolean(model_name, key)


def resolve_torch_device(device):
    """
    Map a CLI --device value to a torch device for tensors and probes.
    'remote' runs the LM on NDIF, so local tensors stay on CPU.
    """
    if device in ("remote", "cpu"):
        return "cpu"
    if str(device).startswith("cuda") and not t.cuda.is_available():
        return "cpu"
    return device


def load_model(model_name, device="remote"):
    print(f"Loading model {model_name}...")
    weights_directory = config[model_name]["weights_directory"]
    if device == "remote":
        return LanguageModel(weights_directory)
    if device == "cpu":
        return LanguageModel(weights_directory, dtype=t.float32, device_map="cpu")
    return LanguageModel(weights_directory, dtype=t.bfloat16, device_map="auto")


def residual_stream(layer_output):
    """
    Normalize traced layer outputs across backends to a [batch, seq, hidden] tensor.
    """
    hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
    return hidden


def last_token_hidden_state(layer_output):
    """
    Return last-token hidden states from a traced layer output.
    """
    hidden = residual_stream(layer_output)
    if hidden.ndim == 3:
        return hidden[:, -1, :]
    if hidden.ndim == 2:
        return hidden
    raise ValueError(
        f"Unsupported traced layer output shape {tuple(hidden.shape)}; expected 2D or 3D tensor."
    )


def materialize(saved):
    """
    nnsight returns saved proxies for remote traces, but local traces may already return tensors.
    """
    return saved.value if hasattr(saved, "value") else saved


def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(os.path.join(ROOT, "datasets", f"{dataset_name}.csv"))
    return dataset["statement"].tolist()


def get_acts(statements, model, layers, remote=True):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    acts = {}
    with model.trace(statements, remote=remote, **TRACER_KWARGS):
        for layer in layers:
            acts[layer] = last_token_hidden_state(model.model.layers[layer].output).save()

    for layer, act in acts.items():
        acts[layer] = materialize(act)

    return acts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate activations for statements in a dataset"
    )
    parser.add_argument(
        "--model",
        default="llama-2-13b",
        help="Model key from config.ini, e.g. llama-2-7b, llama-2-13b, llama-2-70b",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
        help="Layers to save embeddings from. Use -1 for all layers.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Names of datasets, without .csv extension",
    )
    parser.add_argument(
        "--output_dir",
        default="acts",
        help="Directory to save activations to",
    )
    parser.add_argument(
        "--noperiod",
        action="store_true",
        default=False,
        help="Set flag if you don't want to add a period to the end of each statement",
    )
    parser.add_argument(
        "--device",
        default="remote",
        help="NDIF remote, 'cpu', or a torch device such as cuda:0",
    )
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
        save_dir = os.path.join(args.output_dir, args.model)
        if args.noperiod:
            save_dir = os.path.join(save_dir, "noperiod")
        save_dir = os.path.join(save_dir, dataset)
        os.makedirs(save_dir, exist_ok=True)

        for idx in tqdm(range(0, len(statements), 25)):
            acts = get_acts(
                statements[idx : idx + 25],
                model,
                layers,
                args.device == "remote",
            )
            for layer, act in acts.items():
                t.save(act, os.path.join(save_dir, f"layer_{layer}_{idx}.pt"))
