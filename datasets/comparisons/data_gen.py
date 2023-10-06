import pandas as pd
import os

ROOT = '~/statement_reps/datasets'

numbers = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
    'twenty', 'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four', 'twenty-five', 'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine',
    'thirty', 'thirty-one', 'thirty-two', 'thirty-three', 'thirty-four', 'thirty-five', 'thirty-six', 'thirty-seven', 'thirty-eight', 'thirty-nine',
    'forty', 'forty-one', 'forty-two', 'forty-three', 'forty-four', 'forty-five', 'forty-six', 'forty-seven', 'forty-eight', 'forty-nine',
    'fifty', 'fifty-one', 'fifty-two', 'fifty-three', 'fifty-four', 'fifty-five', 'fifty-six', 'fifty-seven', 'fifty-eight', 'fifty-nine',
    'sixty', 'sixty-one', 'sixty-two', 'sixty-three', 'sixty-four', 'sixty-five', 'sixty-six', 'sixty-seven', 'sixty-eight', 'sixty-nine',
    'seventy', 'seventy-one', 'seventy-two', 'seventy-three', 'seventy-four', 'seventy-five', 'seventy-six', 'seventy-seven', 'seventy-eight', 'seventy-nine',
    'eighty', 'eighty-one', 'eighty-two', 'eighty-three', 'eighty-four', 'eighty-five', 'eighty-six', 'eighty-seven', 'eighty-eight', 'eighty-nine',
    'ninety', 'ninety-one', 'ninety-two', 'ninety-three', 'ninety-four', 'ninety-five', 'ninety-six', 'ninety-seven', 'ninety-eight', 'ninety-nine',
]

assert len(numbers) == 100

def is_valid(i, j):
    if i == j:
        return False
    if i < 50 or j < 50:
        return False
    if i % 10 == 0 or j % 10 == 0:
        return False
    return True

larger_out = {
    'statement' : [],
    'label' : [],
    'n1' : [],
    'n2' : [],
    'diff' : [],
    'abs_diff' : [],
}
smaller_out = {
    'statement' : [],
    'label' : [],
    'n1' : [],
    'n2' : [],
    'diff' : [],
    'abs_diff' : [],
}

for i, x in enumerate(numbers):
    for j, y in enumerate(numbers):
        
        if not is_valid(i, j):
            continue
        larger_statement = f'{x.capitalize()} is larger than {y}.'
        smaller_statement = f'{x.capitalize()} is smaller than {y}.'
        if i < j:
            larger_label, smaller_label = 0, 1
        else:
            larger_label, smaller_label = 1, 0
        larger_out['statement'].append(larger_statement), smaller_out['statement'].append(smaller_statement)
        larger_out['label'].append(larger_label), smaller_out['label'].append(smaller_label)
        larger_out['n1'].append(i), smaller_out['n1'].append(i)
        larger_out['n2'].append(j), smaller_out['n2'].append(j)
        larger_out['diff'].append(i - j), smaller_out['diff'].append(i - j)
        larger_out['abs_diff'].append(abs(i - j)), smaller_out['abs_diff'].append(abs(i - j))

larger_df = pd.DataFrame(larger_out)
smaller_df = pd.DataFrame(smaller_out)

larger_df.to_csv(os.path.join(ROOT, 'larger_than.csv'), index=False)
smaller_df.to_csv(os.path.join(ROOT, 'smaller_than.csv'), index=False)