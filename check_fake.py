import pandas as pd
import warnings
warnings.simplefilter('ignore')

total_path = r"D:\RareDisease-traindata\LLLdataset\dataset\processed\total.xlsx"
fake_path = r"D:\RareDisease-traindata\LLLdataset\dataset\processed\FakeDisease.xlsx"

total_df = pd.read_excel(total_path)
targets = total_df[total_df['frequency'] < 15]

fake_df = pd.read_excel(fake_path)
# 统计每种 MONDO 生成的独立病例(case_id)数
fake_counts = fake_df.groupby('mondo_id')['case_id'].nunique().to_dict()

perfect_match = 0
failed_or_skipped = []
for _, row in targets.iterrows():
    m_id = row['mondo_id']
    old_freq = int(row['frequency'])
    needed = 15 - old_freq
    faked = fake_counts.get(m_id, 0)
    
    if needed == faked:
        perfect_match += 1
    else:
        failed_or_skipped.append((m_id, old_freq, faked))

print(f"需要补充的疾病总数: {len(targets)}")
print(f"完美补充到 15 的疾病数: {perfect_match}")
print(f"未能补足的孤岛疾病数: {len(failed_or_skipped)}")

if failed_or_skipped:
    print("\n未能补足的疾病示例 (可能是该疾病在权威库中彻底没有HPO记录导致无法造假):")
    for m, o, f in failed_or_skipped[:10]:
        print(f" - {m}: 原本有 {o} 个, 新造了 {f} 个, 缺口 {15 - o - f} 个")
