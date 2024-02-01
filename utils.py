import torch as t
import pandas as pd
import os
from glob import glob
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25


def get_pcs(X, k=2, offset=0):
    """
    Performs Principal Component Analysis (PCA) on the n x d data matrix X. 
    Returns the k principal components, the corresponding eigenvalues and the projected data.
    """

    # Subtract the mean to center the data
    X = X - t.mean(X, dim=0)
    
    # Compute the covariance matrix
    cov_mat = t.mm(X.t(), X) / (X.size(0) - 1)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = t.linalg.eigh(cov_mat)
    
    # Since the eigenvalues and vectors are not necessarily sorted, we do that now
    sorted_indices = t.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the pcs
    eigenvectors = eigenvectors[:, offset:offset+k]
    
    return eigenvectors

def dict_recurse(d, f):
    """
    Recursively applies a function to a dictionary.
    """
    if isinstance(d, dict):
        out = {}
        for key in d:
            out[key] = dict_recurse(d[key], f)
        return out
    else:
        return f(d)

def collect_acts(dataset_name, model, layer, noperiod=False, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(ROOT, 'acts', model)
    if noperiod:
        directory = os.path.join(directory, 'noperiod')
    directory = os.path.join(directory, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    if len(activation_files) == 0:
        raise ValueError(f"Dataset {dataset_name} not found.")
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt')).to(device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
    acts = t.cat(acts, dim=0).float().to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts

def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    return t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)

class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction
    
    def add_dataset(self, dataset_name, model_size, layer, label='label', split=None, seed=None, noperiod=False, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(dataset_name, model_size, layer, noperiod=noperiod, center=center, scale=scale, device=device)
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset_name}.csv'))
        labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert 0 < split and split < 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels

    def set_pca(self, datasets, k=3, dim_offset=0):
        """
        Sets the projection matrix for dimensionality reduction by doing pca on the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        """
        acts, _ = self.get(datasets, proj=False)
        self.proj = get_pcs(acts, k=k, offset=dim_offset)

        self.data = dict_recurse(self.data, lambda x : (t.mm(x[0], self.proj), x[1]))
    




