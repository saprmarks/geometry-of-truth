# The Geometry of Truth

This repository is associated to the paper [*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*](https://arxiv.org/abs/2310.06824) by Samuel Marks and Max Tegmark. See also our [interactive dataexplorer](https://saprmarks.github.io/geometry-of-truth/dataexplorer).

([View this page on github](https://github.com/saprmarks/geometry-of-truth).)

## Set-up

Python 3.10–3.12 is recommended. Clone the repo and install requirements:

```
git clone git@github.com:saprmarks/geometry-of-truth.git
cd geometry-of-truth
pip install -r requirements.txt
```

Activation collection and the paper experiments use [nnsight](https://nnsight.net/). You can run them locally on a GPU (or CPU, slowly) or remotely on [NDIF](https://ndif.us/) with `--device remote`.

Llama-2 weights on Hugging Face are gated. Accept the license on the model card and log in once:

```
huggingface-cli login
```

`config.ini` already points at the public Hugging Face repo ids. If you store weights on disk, replace `weights_directory` with the **absolute path** to that directory. Hugging Face repo ids are also supported.

Before the notebooks or probes will run, generate activations for the datasets you want. For example:

```
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets cities neg_cities --device cuda:0
```

On a machine without a GPU:

```
python generate_acts.py --model llama-2-7b --layers 12 --datasets cities neg_cities --device cpu
```

On NDIF:

```
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets cities neg_cities --device remote
```

Activations are stored under `acts/`. To save every layer, use `--layers -1`.

## Files

* `dataexplorer.ipynb`: visualizations of the datasets, including figures from the paper. Uses whatever activations you have already generated.
* `generalization.ipynb`: train probes on one dataset and check generalization to another, including the generalization matrix in the paper.
* `few_shot.py`: calibrated 5-shot baseline.
* `generate_acts.py`: extract residual-stream activations with nnsight.
* `interventions.py`: causal intervention experiments from the paper.
* `logprobs.py`: sequence log-probabilities for dataset statements.
* `patching.py` / `patching.ipynb`: activation-patching experiments and plots.
* `probes.py`: logistic, mass-mean, and CCS probe classes.
* `utils.py` and `visualization_utils.py`: dataset loading and PCA plots.
