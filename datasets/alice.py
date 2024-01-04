"""
add Alice/Bob to the start of statements in the given datasets
"""

import pandas as pd
import random


dataset_names = [
    'cities',
    'neg_cities',
]

datasets = [
    pd.read_csv(f'datasets/{dataset_name}.csv') for dataset_name in dataset_names
]

n_statements = len(datasets[0])
assert all(len(dataset) == n_statements for dataset in datasets)

distractor_idxs = []
for idx in range(2):
    distractor_idxs.extend([idx] * (n_statements // 2))
random.shuffle(distractor_idxs)


for dataset, name in zip(datasets, dataset_names):
    statements = []
    has_alices = []
    for (_, row), distractor_idx in zip(dataset.iterrows(), distractor_idxs):
        statement = f"Alice: {row['statement']}" if distractor_idx == 0 else f"Bob: {row['statement']}"
        statements.append(statement)
        has_alices.append(distractor_idx == 0)
    dataset['statement'] = statements
    dataset['has_alice'] = has_alices
    dataset.to_csv(f'datasets/{name}_alice.csv', index=False)