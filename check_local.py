import pandas as pd
import json

# 1. Load missing mondos
no_map_path = r'D:\RareDisease-traindata\LLLdataset\DiseaseHy\no_map.xlsx'
df_detail = pd.read_excel(no_map_path, sheet_name='detail')
missing_mondos = set(df_detail['mondo_id'].dropna().astype(str).unique())
print(f'Total missing unique MONDOs in no_map.xlsx (detail): {len(missing_mondos)}')

# 2. Check mondo.json
mondo_json_path = r'D:\RareDisease-traindata\data\raw_data\mondo.json'
with open(mondo_json_path, 'r', encoding='utf-8') as f:
    mondo_data = json.load(f)

# Mondo json format usually has 'graphs' -> [0] -> 'nodes'
nodes = mondo_data.get('graphs', [{}])[0].get('nodes', [])
local_mondo_ids = set()
for node in nodes:
    node_id = node.get('id', '')
    if 'MONDO_' in node_id or 'MONDO:' in node_id:
        # standardizing to MONDO:xxxx
        std_id = node_id.replace('http://purl.obolibrary.org/obo/MONDO_', 'MONDO:')
        local_mondo_ids.add(std_id)

found_in_mondo_json = missing_mondos.intersection(local_mondo_ids)
print(f'\nFound in local mondo.json: {len(found_in_mondo_json)} out of {len(missing_mondos)}')
if len(found_in_mondo_json) < len(missing_mondos):
    not_found = missing_mondos - found_in_mondo_json
    print(f'Example of MONDOs NOT in mondo.json: {list(not_found)[:5]}')
else:
    print('All missing MONDOs exist in mondo.json!')

# Check hpo_mondo_mapping_report.xlsx
mapping_path = r'D:\RareDisease-traindata\data\raw_data\hpo_mondo_mapping_report.xlsx'
df_map = pd.read_excel(mapping_path)
# we don't know columns exactly, check whole dataframe
mapped_items = set(df_map.values.flatten().tolist())
found_in_mapping = [x for x in missing_mondos if x in mapped_items]
print(f'\nFound {len(found_in_mapping)} in hpo_mondo_mapping_report.xlsx')

