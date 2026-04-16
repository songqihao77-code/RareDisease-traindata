import pandas as pd

file_path = r'D:\RareDisease-traindata\LLLdataset\dataset\rare_disease_hgnn_clean_package_v59\unseen_training_unique_profiles.csv'
df = pd.read_csv(file_path)

cases = df['case_id'].unique()[:3]
for case in cases:
    patient = df[df['case_id'] == case]
    mondo = patient['mondo_label'].iloc[0]
    hpos = patient['hpo_id'].tolist()
    print(f'Case: {case} | Disease: {mondo} | HPO Count: {len(hpos)}')
    print(f'  HPOs: {", ".join(hpos)}\n')
