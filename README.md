# The Geometry of Truth

We include here all code for replicating the figures and experiments of our paper *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*.

## Using LLaMA
You should have your own LLaMA weights stored on the machine containing this repo. There are a few places where you'll need to manually replace the line of code
```
LLAMA_DIRECTORY = '/home/ubuntu/llama_hf
```
with where you store your weights. These places are in: `few_shot.py`, `generate_acts.py`, and `interventions.ipynb`. You'll also need to change the line
```
ROOT = '/home/ubuntu/statement_reps/'
```
in `few_shot.py` and `utils.py` to store the path to this directory (i.e. the pathname should be of the form ".../geometry_of_truth/").

## Generating activations
You'll need to generate the LLaMA activations for the datasets you'd like to work with. You do this with a command like
```
python generate_acts.py --model 13B --layers 8 10 12 --datasets cities neg_cities --device cuda:0
```
These activations will be stored in the acts directory. If you want to save activations for all layers, simply use `--layers -1`.

## Files
This directory contains the following files:
* `dataexplorer.ipynb`: for generating visualizations of the datasets. Code for reproducing figures in the text is included.
* `few_shot.py`: for implementing the calibrated 5-shot baseline.
* `generalization.ipynb`: for training probes on one dataset and checking generalization to another. Includes code for reproducing the generalization matrix in the text.
* `interventions.ipynb`: for reproducing the causal intervention experiments from the text.
* `probes.py`: contains definitions of probe classes.
* `utils.py` and `visualization_utils.py`: utilities for managing datasets and producing visualizations. 