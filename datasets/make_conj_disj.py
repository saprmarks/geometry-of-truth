import pandas as pd
import os
import argparse
import math

ROOT = '~/statement_reps/datasets'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, nargs=2)
parser.add_argument('--datapoints', type=int, default=1500)
args = parser.parse_args()

datasets = args.datasets

dfs = [
    pd.read_csv(os.path.join(ROOT, f'{dataset}.csv')) for dataset in datasets
]

# make conjunctions
p_true = math.sqrt(1/2) # to make the truth of the two statements independent, sample true statements with this probability

ns = {
    (1, 1) : int(p_true ** 2 * args.datapoints),
    (1, 0) : int(p_true * (1 - p_true) * args.datapoints),
    (0, 1) : int(p_true * (1 - p_true) * args.datapoints),
    (0, 0) : int((1 - p_true) ** 2 * args.datapoints)
}

df_out = {
    'statement' : [],
    'label' : []
}
for col in dfs[0].columns:
    df_out[f'{col}1'] = []
for col in dfs[1].columns:
    df_out[f'{col}2'] = []

for (label1, label2), n in ns.items():
    for _ in range(n):
        df1 = dfs[0][dfs[0]['label'] == label1]
        df2 = dfs[1][dfs[1]['label'] == label2]
        row1, row2 = df1.sample(1), df2.sample(1)
        statement1 = row1['statement'].values[0]
        statement2 = row2['statement'].values[0]
        statement = f'It is the case both that {statement1[0].lower()}{statement1[1:-1]} and that {statement2[0].lower()}{statement2[1:-1]}.'
        label = int(bool(label1) and bool(label2))
        df_out['statement'].append(statement)
        df_out['label'].append(label)
        for col in dfs[0].columns:
            df_out[f'{col}1'].append(row1[col].values[0])
        for col in dfs[1].columns:
            df_out[f'{col}2'].append(row2[col].values[0])

df_out = pd.DataFrame(df_out)
df_out.to_csv(os.path.join(ROOT, f'{datasets[0]}_{datasets[1]}_conj.csv'), index=False)

# make disjunctions
p_true = 1 - math.sqrt(1/2) # to make the truth of the two statements independent, sample true statements with this probability

ns = {
    (1, 1) : int(p_true ** 2 * args.datapoints),
    (1, 0) : int(p_true * (1 - p_true) * args.datapoints),
    (0, 1) : int(p_true * (1 - p_true) * args.datapoints),
    (0, 0) : int((1 - p_true) ** 2 * args.datapoints)
}

df_out = {
    'statement' : [],
    'label' : []
}

for col in dfs[0].columns:
    df_out[f'{col}1'] = []
for col in dfs[1].columns:
    df_out[f'{col}2'] = []

for (label1, label2), n in ns.items():
    for _ in range(n):
        df1 = dfs[0][dfs[0]['label'] == label1]
        df2 = dfs[1][dfs[1]['label'] == label2]
        row1, row2 = df1.sample(1), df2.sample(1)
        statement1 = row1['statement'].values[0]
        statement2 = row2['statement'].values[0]
        statement = f'It is the case either that {statement1[0].lower()}{statement1[1:-1]} or that {statement2[0].lower()}{statement2[1:-1]}.'
        label = int(bool(label1) or bool(label2))
        df_out['statement'].append(statement)
        df_out['label'].append(label)
        for col in dfs[0].columns:
            df_out[f'{col}1'].append(row1[col].values[0])
        for col in dfs[1].columns:
            df_out[f'{col}2'].append(row2[col].values[0])

df_out = pd.DataFrame(df_out)
df_out.to_csv(os.path.join(ROOT, f'{datasets[0]}_{datasets[1]}_disj.csv'), index=False)