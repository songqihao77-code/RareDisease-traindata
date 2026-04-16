"""深度分析HMS和mimic_test数据集低准确率的原因"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.build_hypergraph import load_disease_incidence

test_dir = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/test')
train_dir = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/train')

# 加载HPO和Disease索引
hpo_df = pd.read_excel(r'D:/RareDisease-traindata/LLLdataset/DiseaseHy/processed/HPO_index_v4.xlsx', dtype={'hpo_id': str})
hpo_set = set(hpo_df['hpo_id'].tolist())
hpo_to_idx = dict(zip(hpo_df['hpo_id'], hpo_df['hpo_idx']))

disease_df = pd.read_excel(r'D:/RareDisease-traindata/LLLdataset/DiseaseHy/processed/Disease_index_v4.xlsx', dtype={'mondo_id': str})
disease_set = set(disease_df['mondo_id'].tolist())
disease_to_idx = dict(zip(disease_df['mondo_id'], disease_df['disease_idx']))

# 加载超图
H_disease = load_disease_incidence(r'D:/RareDisease-traindata/LLLdataset/DiseaseHy/processed/DiseaseHyperedge_sparse_triplets_v4.npz')
H_dense = H_disease.toarray()

print(f'HPO index total: {len(hpo_set)}')
print(f'Disease index total: {len(disease_set)}')
print(f'H_disease shape: {H_disease.shape}')
print('='*100)

def read_file(path):
    if path.suffix == '.csv':
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)

# 收集所有训练集的疾病和HPO
all_train_diseases = set()
all_train_hpos = set()
train_disease_counts = {}
mimic_train_path = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/mimic_rag_0425.csv')
train_files = list(train_dir.iterdir()) 
if mimic_train_path.exists():
    train_files.append(mimic_train_path)

for f in train_files:
    if f.name.startswith('~'): continue
    df = read_file(f)
    diseases_in_file = set(df['mondo_label'].dropna().unique())
    hpos_in_file = set(df['hpo_id'].dropna().unique())
    all_train_diseases.update(diseases_in_file)
    all_train_hpos.update(hpos_in_file)
    for d in diseases_in_file:
        train_disease_counts[d] = train_disease_counts.get(d, 0) + df[df['mondo_label']==d]['case_id'].nunique()

print(f'Training set: {len(all_train_diseases)} unique diseases, {len(all_train_hpos)} unique HPOs')
print('='*100)

# 分析每个测试集
results = {}
for f in sorted(test_dir.iterdir()):
    if f.name.startswith('~'): continue
    df = read_file(f)
    name = f.stem
    
    n_cases = df['case_id'].nunique()
    
    # 每个case的HPO数量
    case_hpo_counts = df.groupby('case_id')['hpo_id'].apply(lambda x: x.dropna().nunique())
    
    # 疾病标签
    case_labels = df.groupby('case_id')['mondo_label'].first()
    unique_diseases = case_labels.nunique()
    
    # 训练集中出现过的疾病
    diseases_seen_in_train = sum(1 for d in case_labels.unique() if d in all_train_diseases)
    diseases_not_in_train = sum(1 for d in case_labels.unique() if d not in all_train_diseases)
    
    # 训练集中该疾病的训练样本数
    disease_train_sample_counts = [train_disease_counts.get(d, 0) for d in case_labels.values]
    
    # 分析每个case的HPO与知识图谱的overlap
    overlap_ratios = []
    for case_id, group in df.groupby('case_id'):
        case_hpos = set(group['hpo_id'].dropna().unique())
        case_hpos_valid = [h for h in case_hpos if h in hpo_to_idx]
        mondo_id = group['mondo_label'].iloc[0]
        
        if mondo_id in disease_to_idx:
            d_idx = disease_to_idx[mondo_id]
            # 知识图谱里该疾病关联的HPO
            kg_hpos_idx = np.where(H_dense[:, d_idx] > 0)[0]
            kg_hpos = set(hpo_df.iloc[kg_hpos_idx]['hpo_id'].tolist())
            
            # 重叠率
            if len(case_hpos_valid) > 0:
                overlap = len(set(case_hpos_valid) & kg_hpos)
                precision = overlap / len(case_hpos_valid) if len(case_hpos_valid) > 0 else 0
                recall = overlap / len(kg_hpos) if len(kg_hpos) > 0 else 0
                overlap_ratios.append({
                    'case_id': case_id,
                    'case_hpo_count': len(case_hpos_valid),
                    'kg_hpo_count': len(kg_hpos),
                    'overlap': overlap,
                    'precision': precision,
                    'recall': recall,
                    'mondo_id': mondo_id
                })
    
    overlap_df = pd.DataFrame(overlap_ratios)
    
    print(f'\n=== {name} ===')
    print(f'  Cases: {n_cases}')
    print(f'  Unique diseases: {unique_diseases}')
    print(f'  Diseases in train: {diseases_seen_in_train}/{unique_diseases} ({diseases_seen_in_train/max(unique_diseases,1)*100:.1f}%)')
    print(f'  Diseases NOT in train: {diseases_not_in_train}')
    print(f'  HPO/case: mean={case_hpo_counts.mean():.1f}, median={case_hpo_counts.median():.1f}, min={case_hpo_counts.min()}, max={case_hpo_counts.max()}')
    print(f'  Train samples per disease: mean={np.mean(disease_train_sample_counts):.1f}, median={np.median(disease_train_sample_counts):.1f}')
    
    if not overlap_df.empty:
        print(f'  KG overlap precision (case HPOs in KG): mean={overlap_df["precision"].mean():.3f}, median={overlap_df["precision"].median():.3f}')
        print(f'  KG overlap recall (KG HPOs in case): mean={overlap_df["recall"].mean():.3f}, median={overlap_df["recall"].median():.3f}')
        print(f'  KG disease HPO count: mean={overlap_df["kg_hpo_count"].mean():.1f}, median={overlap_df["kg_hpo_count"].median():.1f}')
        print(f'  Overlap count: mean={overlap_df["overlap"].mean():.1f}, median={overlap_df["overlap"].median():.1f}')
    
    results[name] = {
        'n_cases': n_cases,
        'unique_diseases': unique_diseases,
        'diseases_in_train': diseases_seen_in_train,
        'hpo_per_case_mean': case_hpo_counts.mean(),
        'overlap_df': overlap_df
    }

# 额外分析：mimic_test的训练数据来源
print('\n' + '='*100)
print('\n=== Detailed: mimic_test vs mimic_train overlap ===')
if mimic_train_path.exists():
    mimic_train = pd.read_csv(mimic_train_path, dtype=str)
    mimic_test = pd.read_csv(test_dir / 'mimic_test.csv', dtype=str)
    
    train_diseases = set(mimic_train['mondo_label'].dropna().unique())
    test_diseases = set(mimic_test['mondo_label'].dropna().unique())
    
    print(f'  mimic_train diseases: {len(train_diseases)}')
    print(f'  mimic_test diseases: {len(test_diseases)}')
    print(f'  Overlap: {len(train_diseases & test_diseases)}')
    print(f'  Test-only diseases: {len(test_diseases - train_diseases)}')
    print(f'  Ratio of test diseases seen in train: {len(train_diseases & test_diseases)/len(test_diseases)*100:.1f}%')

# HMS详细分析
print('\n=== Detailed: HMS disease analysis ===')
hms_test = read_file(test_dir / 'HMS.xlsx')
hms_diseases = hms_test.groupby('case_id')['mondo_label'].first()
for d in hms_diseases.unique():
    in_train = 'YES' if d in all_train_diseases else 'NO'
    train_count = train_disease_counts.get(d, 0)
    
    # 该疾病在KG中有多少HPO
    if d in disease_to_idx:
        d_idx = disease_to_idx[d]
        kg_hpo_count = int(np.sum(H_dense[:, d_idx] > 0))
    else:
        kg_hpo_count = 0
    
    case_count = sum(1 for x in hms_diseases if x == d)
    print(f'  {d}: cases={case_count}, in_train={in_train}, train_samples={train_count}, KG_HPOs={kg_hpo_count}')
