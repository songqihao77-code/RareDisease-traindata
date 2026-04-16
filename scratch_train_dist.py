import pandas as pd
from pathlib import Path

mimic_train = pd.read_csv(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\mimic_rag_0425.csv', dtype=str)
print(f'mimic_train rows: {len(mimic_train)}')
print(f'mimic_train cases: {mimic_train["case_id"].nunique()}')
print(f'mimic_train diseases: {mimic_train["mondo_label"].nunique()}')
print(f'mimic_train HPOs: {mimic_train["hpo_id"].nunique()}')

train_dir = Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\train')
total = 0
for f in sorted(train_dir.iterdir()):
    if f.name.startswith('~'): continue
    if f.suffix == '.csv':
        df = pd.read_csv(f, dtype=str)
    else:
        df = pd.read_excel(f, dtype=str)
    n = df['case_id'].nunique()
    total += n
    print(f'{f.stem}: {n} cases, {len(df)} rows')

fake = Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\FakeDisease.xlsx')
n_fake = 0
if fake.exists():
    df = pd.read_excel(fake, dtype=str)
    n_fake = df['case_id'].nunique()
    print(f'FakeDisease: {n_fake} cases, {len(df)} rows')

n_mimic = mimic_train['case_id'].nunique()
total += n_mimic + n_fake
print(f'\nTotal train cases: {total}')
print(f'mimic ratio: {n_mimic/total*100:.1f}%')
