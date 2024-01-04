import pandas as pd

likely = pd.read_csv('datasets/likely.csv')

n_statements = 1000

statements = []

for _, row in likely.iterrows():
    if row['label'] == 0:
        continue
    statements.append(row['statement'])
    if len(statements) == n_statements:
        break

trues = [1] * (n_statements // 2) + [0] * (n_statements // 2)
bananas = [1] * (n_statements // 4) + [0] * (n_statements // 4) + [1] * (n_statements // 4) + [0] * (n_statements // 4)

for idx, (statement, true, banana) in enumerate(zip(statements, trues, bananas)):
    statements[idx] = statement + (' true' if true else ' false') + (' banana' if banana else ' shed')

df = pd.DataFrame({
    'statement': statements,
    'true': trues,
    'banana': bananas,
    'xor': [true ^ banana for true, banana in zip(trues, bananas)]
})

df.to_csv('datasets/xor.csv', index=False)
