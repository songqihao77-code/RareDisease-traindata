import pandas as pd
import numpy as np
from scipy.sparse import load_npz

old_file = r'D:\RareDisease-traindata\LLLdataset\dataset\rare_disease_hgnn_training_package_v6\all_diseases_5_to_15_profiles_minimal.xlsx'
new_file = r'D:\RareDisease-traindata\LLLdataset\dataset\rare_disease_hgnn_clean_package_v59\unseen_training_unique_profiles.csv'
v58_path = r'D:\RareDisease-traindata\LLLdataset\DiseaseHy\v58_sparse_gap_review_package\v58DiseaseHy.npz'
v59_path = r'D:\RareDisease-traindata\LLLdataset\dataset\rare_disease_hgnn_clean_package_v59\v59DiseaseHy.npz'

def analyze_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        num_cases = df['case_id'].nunique()
        num_diseases = df['mondo_label'].nunique()
        avg_hpo = len(df) / num_cases
        return f'Cases: {num_cases}, Unique Diseases: {num_diseases}, Avg HPOs/Case: {avg_hpo:.2f}'
    except Exception as e:
        return f'Error loading {file_path}: {e}'

def analyze_npz(file_path):
    try:
        npz = np.load(file_path, allow_pickle=False)
        keys = set(npz.keys())
        row_key = next((k for k in ('rows', 'row') if k in keys), None)
        col_key = next((k for k in ('cols', 'col') if k in keys), None)
        value_key = next((k for k in ('vals', 'data') if k in keys), None)
        shape = tuple(npz['shape'])
        nnz = len(npz[value_key])
        return f'Shape: {shape}, Non-zeros: {nnz}'
    except Exception as e:
        return f'Error loading npz: {e}'

print('--- Old Synthetic Profiles ---')
print(analyze_data(old_file))
print('\n--- New Unseen Training Profiles ---')
print(analyze_data(new_file))

print('\n--- v58 Disease Incidence ---')
print(analyze_npz(v58_path))
print('\n--- v59 Disease Incidence ---')
print(analyze_npz(v59_path))

