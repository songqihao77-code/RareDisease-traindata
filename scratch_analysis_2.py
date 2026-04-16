"""
Optimized deep analysis: HPO specificity, noise ratio, skip confusion (too slow).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from src.data.build_hypergraph import load_disease_incidence

test_dir = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/test')
train_dir = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/train')

hpo_df = pd.read_excel(r'D:/RareDisease-traindata/LLLdataset/DiseaseHy/processed/HPO_index_v4.xlsx', dtype={'hpo_id': str})
hpo_set = set(hpo_df['hpo_id'].tolist())
hpo_to_idx = dict(zip(hpo_df['hpo_id'], hpo_df['hpo_idx']))
idx_to_hpo = {v: k for k, v in hpo_to_idx.items()}

disease_df = pd.read_excel(r'D:/RareDisease-traindata/LLLdataset/DiseaseHy/processed/Disease_index_v4.xlsx', dtype={'mondo_id': str})
disease_to_idx = dict(zip(disease_df['mondo_id'], disease_df['disease_idx']))

H_disease = load_disease_incidence(r'D:/RareDisease-traindata/LLLdataset/DiseaseHy/processed/DiseaseHyperedge_sparse_triplets_v4.npz')
H_dense = H_disease.toarray()

hpo_degrees = np.array(H_disease.sum(axis=1)).flatten()

def read_file(path):
    if path.suffix == '.csv':
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)

# Load all training data HPO freq
all_train_files = list(train_dir.iterdir())
mimic_train_path = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/mimic_rag_0425.csv')
if mimic_train_path.exists():
    all_train_files.append(mimic_train_path)
fake_path = Path(r'D:/RareDisease-traindata/LLLdataset/dataset/processed/FakeDisease.xlsx')
if fake_path.exists():
    all_train_files.append(fake_path)

train_hpo_counter = Counter()
for f in all_train_files:
    if f.name.startswith('~'): continue
    df = read_file(f)
    if 'hpo_id' in df.columns:
        for h in df['hpo_id'].dropna():
            if h in hpo_set:
                train_hpo_counter[h] += 1

datasets = {}
for f in sorted(test_dir.iterdir()):
    if f.name.startswith('~'): continue
    datasets[f.stem] = read_file(f)

print("="*100)
print("PART 1: HPO specificity analysis")
print("="*100)

for name, df in datasets.items():
    hpos_used = df['hpo_id'].dropna().unique()
    valid_hpos = [h for h in hpos_used if h in hpo_to_idx]
    if not valid_hpos:
        continue
    
    degrees = [hpo_degrees[hpo_to_idx[h]] for h in valid_hpos]
    train_freqs = [train_hpo_counter.get(h, 0) for h in valid_hpos]
    rare_hpos = [h for h in valid_hpos if train_hpo_counter.get(h, 0) == 0]
    
    print(f'\n--- {name} ---')
    print(f'  Unique HPOs used: {len(valid_hpos)}')
    print(f'  HPO KG degree: mean={np.mean(degrees):.1f}, median={np.median(degrees):.1f}')
    print(f'  HPOs NOT seen in training: {len(rare_hpos)} ({len(rare_hpos)/len(valid_hpos)*100:.1f}%)')
    print(f'  HPO train freq: mean={np.mean(train_freqs):.1f}, median={np.median(train_freqs):.1f}')
    
    high_degree = sum(1 for d in degrees if d > 500)
    low_degree = sum(1 for d in degrees if d < 10)
    print(f'  HPOs degree>500 (nonspecific): {high_degree} ({high_degree/len(degrees)*100:.1f}%)')
    print(f'  HPOs degree<10 (highly specific): {low_degree} ({low_degree/len(degrees)*100:.1f}%)')

print("\n" + "="*100)
print("PART 2: Per-case noise analysis")
print("="*100)

for name in ['HMS', 'mimic_test', 'RAMEDIS', 'MyGene2', 'DDD', 'MME', 'LIRICAL']:
    df = datasets[name]
    
    noise_ratios = []
    signal_counts = []
    noise_counts = []
    total_hpo_counts = []
    
    for case_id, group in df.groupby('case_id'):
        mondo_id = group['mondo_label'].iloc[0]
        case_hpos = [h for h in group['hpo_id'].dropna().unique() if h in hpo_to_idx]
        
        if not case_hpos or mondo_id not in disease_to_idx:
            continue
        
        d_idx = disease_to_idx[mondo_id]
        kg_hpos_mask = H_dense[:, d_idx] > 0
        
        signal = sum(1 for h in case_hpos if kg_hpos_mask[hpo_to_idx[h]])
        noise = len(case_hpos) - signal
        
        noise_ratios.append(noise / len(case_hpos))
        signal_counts.append(signal)
        noise_counts.append(noise)
        total_hpo_counts.append(len(case_hpos))
    
    if noise_ratios:
        print(f'\n--- {name} ---')
        print(f'  Total HPO/case: mean={np.mean(total_hpo_counts):.1f}')
        print(f'  Signal HPO/case: mean={np.mean(signal_counts):.1f}, median={np.median(signal_counts):.1f}')
        print(f'  Noise HPO/case: mean={np.mean(noise_counts):.1f}, median={np.median(noise_counts):.1f}')
        print(f'  Noise ratio: mean={np.mean(noise_ratios):.3f}, median={np.median(noise_ratios):.3f}')
        print(f'  Cases with 0 signal: {sum(1 for s in signal_counts if s == 0)} ({sum(1 for s in signal_counts if s == 0)/len(signal_counts)*100:.1f}%)')
        print(f'  Cases with noise>90%: {sum(1 for r in noise_ratios if r > 0.9)} ({sum(1 for r in noise_ratios if r > 0.9)/len(noise_ratios)*100:.1f}%)')
        print(f'  Cases with noise>80%: {sum(1 for r in noise_ratios if r > 0.8)} ({sum(1 for r in noise_ratios if r > 0.8)/len(noise_ratios)*100:.1f}%)')

print("\n" + "="*100)
print("PART 3: Top-50 common HPO ratio")
print("="*100)

top50_hpos = set()
top50_idx = hpo_degrees.argsort()[-50:][::-1]
for idx in top50_idx:
    top50_hpos.add(idx_to_hpo[idx])

for name in ['HMS', 'mimic_test', 'RAMEDIS', 'MyGene2', 'DDD', 'MME', 'LIRICAL']:
    df = datasets[name]
    common_ratios = []
    for case_id, group in df.groupby('case_id'):
        case_hpos = set(group['hpo_id'].dropna().unique()) & hpo_set
        if not case_hpos:
            continue
        common = len(case_hpos & top50_hpos)
        common_ratios.append(common / len(case_hpos))
    
    print(f'  {name}: top-50 HPO ratio mean={np.mean(common_ratios):.3f}, median={np.median(common_ratios):.3f}')

print("\n" + "="*100)
print("PART 4: Disease confusion analysis (vectorized)")
print("="*100)

# Vectorized Jaccard: for target diseases, find max Jaccard with any other disease
# Use sparse matrix operations
from scipy.sparse import csc_matrix

H_csc = csc_matrix(H_disease)

def get_max_jaccards_for_diseases(target_mondo_ids, top_n=5):
    """Compute max Jaccard for a set of target diseases against all."""
    results = []
    for mondo_id in target_mondo_ids:
        if mondo_id not in disease_to_idx:
            continue
        d_idx = disease_to_idx[mondo_id]
        target_col = H_dense[:, d_idx]
        target_set = set(np.where(target_col > 0)[0])
        n_target = len(target_set)
        if n_target == 0:
            continue
        
        # Dot product gives intersection size
        intersections = H_dense.T @ target_col  # shape: [num_disease]
        intersections[d_idx] = 0  # exclude self
        
        # Union = |A| + |B| - |A∩B|
        disease_sizes = np.array(H_disease.sum(axis=0)).flatten()
        unions = n_target + disease_sizes - intersections
        unions = np.maximum(unions, 1)
        
        jaccards = intersections / unions
        top_idx = np.argsort(jaccards)[-top_n:][::-1]
        
        results.append({
            'mondo_id': mondo_id,
            'max_jaccard': jaccards[top_idx[0]],
            'mean_top5_jaccard': np.mean(jaccards[top_idx]),
            'n_confusable_j03': int(np.sum(jaccards > 0.3)),
            'n_confusable_j05': int(np.sum(jaccards > 0.5)),
            'kg_hpo_count': n_target,
        })
    return pd.DataFrame(results)

for name in ['HMS', 'mimic_test', 'RAMEDIS', 'MyGene2', 'DDD', 'MME', 'LIRICAL']:
    df = datasets[name]
    case_labels = df.groupby('case_id')['mondo_label'].first()
    unique_diseases = list(case_labels.unique())
    
    amb_df = get_max_jaccards_for_diseases(unique_diseases)
    if not amb_df.empty:
        print(f'\n--- {name} ({len(amb_df)} diseases) ---')
        print(f'  Max Jaccard: mean={amb_df["max_jaccard"].mean():.3f}, median={amb_df["max_jaccard"].median():.3f}')
        print(f'  Top-5 Jaccard: mean={amb_df["mean_top5_jaccard"].mean():.3f}')
        print(f'  Diseases with confounders (J>0.3): mean={amb_df["n_confusable_j03"].mean():.1f}')
        print(f'  Diseases with confounders (J>0.5): mean={amb_df["n_confusable_j05"].mean():.1f}')
        print(f'  KG HPO count: mean={amb_df["kg_hpo_count"].mean():.1f}')
