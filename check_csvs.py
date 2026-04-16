import pandas as pd
import os

base = r'D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed'
files = ['HPOA.csv', 'orphanet_mondo_hpo_raw_weight.csv', 'GARD.csv']

for f in files:
    path = os.path.join(base, f)
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=5)
        print(f"\n--- {f} ---")
        print("Columns:", list(df.columns))
        print(df.head(2).to_dict(orient='records'))
    else:
        print(f"{f} not found!")
