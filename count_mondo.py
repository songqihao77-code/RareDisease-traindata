import pandas as pd
from collections import Counter
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

files = [
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\DDD.xlsx",
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\HMS.xlsx",
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\LIRICAL.xlsx",
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\mimic-rare(law).xlsx",
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\MME.xlsx",
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\MyGene2.xlsx",
    r"D:\RareDisease-traindata\LLLdataset\dataset\processed\RAMEDIS.xlsx"
]

all_mondos = []

for f in files:
    if not os.path.exists(f):
        print(f"Warning: {f} does not exist.")
        continue
    try:
        df = pd.read_excel(f)
        mondo_cols = [c for c in df.columns if 'mondo' in str(c).lower()]
        if mondo_cols:
            col = mondo_cols[0]
            mondos = df[col].dropna().astype(str).tolist()
            mondos = [m.strip() for m in mondos if 'MONDO' in m.upper()]
            all_mondos.extend(mondos)
            print(f"Processed {os.path.basename(f)}, extracted {len(mondos)} MONDO tags.")
        else:
             print(f"Error: No mondo column found in {os.path.basename(f)}")
    except Exception as e:
        print(f"Failed to process {f}: {e}")

counter = Counter(all_mondos)
out_df = pd.DataFrame(counter.items(), columns=['mondo_id', 'frequency'])
out_df = out_df.sort_values(by='frequency', ascending=False)

out_path = r"D:\RareDisease-traindata\LLLdataset\dataset\processed\total.xlsx"
out_df.to_excel(out_path, index=False)
print(f"\nAggregation complete. Saved total.xlsx with {len(out_df)} unique diseases.")
