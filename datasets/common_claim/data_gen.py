import pandas as pd
import os

ROOT = '~/statement_reps/datasets'

def is_valid(row):
    if row['agreement'] != 1:
        return False
    if row['label'] == 'Neither':
        return False
    return True

df = pd.read_csv(os.path.join(ROOT, 'common_claim.csv'))

df = df[df.apply(is_valid, axis=1)]

n_true, n_false = len(df[df['label'] == 'True']), len(df[df['label'] == 'False'])
n = min([n_true, n_false])

df_true = df[df['label'] == 'True'].sample(n=n)
df_false = df[df['label'] == 'False'].sample(n=n)
df_out = pd.concat([df_true, df_false])
df_out = df_out.drop(columns=['agreement'])
df_out = df_out.drop(columns=['Unnamed: 0'])

df_out['label'] = df_out['label'].apply(lambda x: 1 if x == 'True' else 0)
df_out = df_out.rename(columns={'examples': 'statement'})

df_out.to_csv(os.path.join(ROOT, 'common_claim_true_false.csv'), index=False)