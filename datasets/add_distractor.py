"""
add distractor word to the end of each statement in the given datasets
"""

import pandas as pd
import random

distractors = [
    'banana',
    'shed',
]

dataset_names = [
    'cities',
    'neg_cities',
]

n_distractors = len(distractors)
datasets = [
    pd.read_csv(f'datasets/{dataset_name}.csv') for dataset_name in dataset_names
]

n_statements = len(datasets[0])
assert all(len(dataset) == n_statements for dataset in datasets)

distractor_idxs = []
for idx in range(n_distractors):
    distractor_idxs.extend([idx] * (n_statements // n_distractors))
random.shuffle(distractor_idxs)

for dataset, name in zip(datasets, dataset_names):
    # truncate the dataset if necessary
    dataset = dataset[:(n_statements // n_distractors) * n_distractors]
    dataset['distractor'] = [distractors[idx] for idx in distractor_idxs]
    dataset['statement'] = [
        statement + f' {distractor}' for statement, distractor in zip(dataset['statement'], dataset['distractor'])
    ]
    dataset.to_csv(f'datasets/{name}_distractor.csv', index=False)