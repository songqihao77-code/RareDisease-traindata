import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

hpoa_path = r'D:\RareDisease-traindata\data\raw_data\phenotype.hpoa'
no_map_path = r'D:\RareDisease-traindata\LLLdataset\DiseaseHy\no_map.xlsx'

print('Loading phenotype.hpoa...')
try:
    hpoa_df = pd.read_csv(hpoa_path, sep='\t', comment='#', dtype=str)
    # Check column names
    db_col = 'database_id' if 'database_id' in hpoa_df.columns else hpoa_df.columns[0]
    hpoa_mondos = set(hpoa_df[db_col].dropna().unique())
    print(f'HPOA loaded. Count of unique database_ids: {len(hpoa_mondos)}')
    print(f'HPOA ID examples: {list(hpoa_mondos)[:3]}')
except Exception as e:
    print('Failed to read HPOA:', e)
    hpoa_mondos = set()

print('\nLoading no_map.xlsx...')
try:
    no_map_df = pd.read_excel(no_map_path, header=None)
    items = no_map_df.values.flatten().tolist()
    missing_mondos = set([str(x).strip() for x in items if pd.notna(x) and str(x).startswith('MONDO')])
    if not missing_mondos:
        missing_mondos = set([str(x).strip() for x in items if pd.notna(x)])
    print(f'no_map loaded. Total unique missing IDs: {len(missing_mondos)}')
    print(f'Missing ID examples: {list(missing_mondos)[:3]}')
except Exception as e:
    print('Failed to read no_map.xlsx:', e)
    missing_mondos = set()

if hpoa_mondos and missing_mondos:
    exact_match = missing_mondos.intersection(hpoa_mondos)
    print(f'\n[Result] Exact matches found in HPOA: {len(exact_match)} out of {len(missing_mondos)}')
    
    missing_ids_only = set([x.split(':')[-1] for x in missing_mondos if ':' in x])
    hpoa_ids_only = set([x.split(':')[-1] for x in hpoa_mondos if ':' in x])
    fuzzy_match = missing_ids_only.intersection(hpoa_ids_only)
    print(f'[Result] Fuzzy matches (by numbers only): {len(fuzzy_match)} out of {len(missing_mondos)}')
