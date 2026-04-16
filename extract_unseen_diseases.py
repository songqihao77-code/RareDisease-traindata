import pandas as pd
import json
import os
from pathlib import Path

# 训练集基础路径 (剔除用户指定的在测试范围内的文件)
TRAIN_DIR = Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\train')

# 映射文件路径
MONDO_JSON = r'D:\RareDisease-traindata\data\raw_data\mondo-base.json'
OMIM_TSV = r'D:\DeepRare-data\mondo_exactmatch_omim.sssom.tsv'
ORPHA_TSV = r'D:\DeepRare-data\mondo_hasdbxref_orphanet.sssom.tsv'

# 输出路径
OUTPUT_DIR = Path(r'D:\新建文件夹')
OUTPUT_FILE = OUTPUT_DIR / 'unseen_diseases_v2.csv'

# 用户指定的 8 个目标数据集 (由于含有不同拓展名，分别直接读取)
TARGET_FILES = [
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\HMS.xlsx'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\LIRICAL.xlsx'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\mimic_rag_0425.csv'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\MME.xlsx'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\MyGene2.xlsx'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\RAMEDIS.xlsx'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\ddd_test.csv'),
    Path(r'D:\RareDisease-traindata\LLLdataset\dataset\processed\test\mimic_test.csv')
]

def read_file(path):
    if path.suffix == '.csv':
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)

print("1. Collecting pure training diseases...")
train_diseases = set()
# 只把纯粹的 train 目录视为已知训练集
for f in TRAIN_DIR.iterdir():
    if f.name.startswith('~') or f.is_dir(): continue
    try:
        df = read_file(f)
        if 'mondo_label' in df.columns:
            train_diseases.update(df['mondo_label'].dropna().unique())
    except Exception as e:
        print(f"Skipped {f.name}: {e}")

print(f"Total pure train diseases: {len(train_diseases)}")

print("2. Collecting target datasets...")
all_unseen = set()

for tf in TARGET_FILES:
    if tf.exists():
        try:
            df = read_file(tf)
            if 'mondo_label' in df.columns:
                file_diseases = set(df['mondo_label'].dropna().unique())
                unseen = file_diseases - train_diseases
                print(f"  {tf.name}: found {len(unseen)} unseen out of {len(file_diseases)} total diseases.")
                for d in unseen:
                    all_unseen.add((d, tf.name))
            else:
                print(f"  {tf.name}: No 'mondo_label' column found.")
        except Exception as e:
            print(f"  Error reading {tf.name}: {e}")
    else:
        print(f"  [WARNING] File not found: {tf}")

print(f"Total collective unseen disease occurrences across the 8 files: {len(all_unseen)}")

# Create mapping dictionary
unseen_dict = {}
for item in all_unseen:
    mondo_id = item[0]
    source = item[1]
    if mondo_id not in unseen_dict:
        unseen_dict[mondo_id] = {'dataset': [source], 'name': 'Unknown'}
    else:
        unseen_dict[mondo_id]['dataset'].append(source)

print("3. Matching names from knowledge graphs...")
for tsv_file in [OMIM_TSV, ORPHA_TSV]:
    if os.path.exists(tsv_file):
        print(f"  Reading {tsv_file}...")
        df_tsv = pd.read_csv(tsv_file, sep='\\t', dtype=str)
        if 'subject_id' in df_tsv.columns and 'subject_label' in df_tsv.columns:
            for _, row in df_tsv.iterrows():
                mondo_id = row['subject_id']
                label = row['subject_label']
                if pd.notna(label) and mondo_id in unseen_dict:
                    if unseen_dict[mondo_id]['name'] == 'Unknown':
                        unseen_dict[mondo_id]['name'] = label

if os.path.exists(MONDO_JSON):
    print(f"  Reading {MONDO_JSON}...")
    with open(MONDO_JSON, 'r', encoding='utf-8') as f:
        mondo_data = json.load(f)
    if 'graphs' in mondo_data and len(mondo_data['graphs']) > 0:
        nodes = mondo_data['graphs'][0].get('nodes', [])
        for node in nodes:
            node_id = node.get('id', '')
            lbl = node.get('lbl')
            if not lbl: continue
            
            if node_id.startswith('http://purl.obolibrary.org/obo/MONDO_'):
                mondo_str = node_id.replace('http://purl.obolibrary.org/obo/MONDO_', 'MONDO:')
                if mondo_str in unseen_dict:
                    if unseen_dict[mondo_str]['name'] == 'Unknown':
                        unseen_dict[mondo_str]['name'] = lbl
            elif node_id in unseen_dict:
                if unseen_dict[node_id]['name'] == 'Unknown':
                    unseen_dict[node_id]['name'] = lbl

print("4. Generating CSV...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []
for mondo_id, data in unseen_dict.items():
    # Join multiple dataset names if the same disease is unseen in multiple sets
    dataset_col = " | ".join(sorted(data['dataset']))
    results.append({
        'mondo_id': mondo_id,
        'dataset_source': dataset_col,
        'disease_name': data['name']
    })

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by=['dataset_source', 'mondo_id']).reset_index(drop=True)
res_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\nDone! Extracted {len(res_df)} unique unseen diseases.")
print(f"Saved correctly mapped file to: {OUTPUT_FILE}")
