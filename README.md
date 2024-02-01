# The Geometry of Truth

This repository is associated to the paper <a href="https://arxiv.org/abs/2310.06824">*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*</a> by Samuel Marks and Max Tegmark. See also our <a href="https://saprmarks.github.io/geometry-of-truth/dataexplorer">interactive dataexplorer</a>.

(<a href="https://github.com/saprmarks/geometry-of-truth">View this page on github</a>.)

## Set-up

Navigate to the location that you want to clone this repo to, clone and enter the repo, and install requirements.
```
git clone git@github.com:saprmarks/geometry-of-truth.git
cd geometry-of-truth
pip install -r requirements.txt
```
Before doing anything, you'll need to generate activations for the datasets. You should have your own LLaMA weights stored on the machine where you cloned this repo. Put the absolute path for the directory containing your LLaMA weights in the file `config.ini`; Huggingface repos are also supported. 

Once that's done, you can generate the LLaMA activations for the datasets you'd like to work with with a command like
```
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets cities neg_cities --device cuda:0
```
These activations will be stored in the acts directory. If you want to save activations for all layers, simply use `--layers -1`.

## Files
This directory contains the following files:
* `dataexplorer.ipynb`: for generating visualizations of the datasets. Code for reproducing figures in the text is included.
* `few_shot.py`: for implementing the calibrated 5-shot baseline.
* `generalization.ipynb`: for training probes on one dataset and checking generalization to another. Includes code for reproducing the generalization matrix in the text.
* `interventions.py`: for reproducing the causal intervention experiments from the text.
* `probes.py`: contains definitions of probe classes.
* `utils.py` and `visualization_utils.py`: utilities for managing datasets and producing visualizations. 


