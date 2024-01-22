import pandas as pd

likely = pd.read_csv('datasets/likely.csv')


statements = []
labels = []

for _, row in likely.iterrows():
    statements.append(row['statement'])
    labels.append(row['label'] == 1)

n = (len(statements) // 4) * 4
statements = statements[:n]
labels = labels[:n]

trues = [True] * (n // 2) + [False] * (n // 2)
bananas = [True] * (n // 4) + [False] * (n // 4) + [True] * (n // 4) + [False] * (n // 4)

for idx, (statement, true, banana) in enumerate(zip(statements, trues, bananas)):
    statements[idx] = statement + (' true' if true else ' false') + (' banana' if banana else ' shed')

df = pd.DataFrame({
    'statement': statements,
    'true': trues,
    'banana': bananas,
    'label': labels,
    'has_true xor has_banana': [true ^ banana for true, banana in zip(trues, bananas)],
    'has_true xor has label' : [true ^ label for true, label in zip(trues, labels)],
    'has_banana xor has label' : [banana ^ label for banana, label in zip(bananas, labels)],
    'has_true xor has_banana xor has_label' : [true ^ banana ^ label for true, banana, label in zip(trues, bananas, labels)],
})

df.to_csv('datasets/xor_new.csv', index=False)
