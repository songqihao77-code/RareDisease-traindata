# mimic_test gold disease - HPO coverage audit

- disease-HPO resource: D:\RareDisease-traindata\LLLdataset\DiseaseHy\rare_disease_hgnn_clean_package_v59\v59_hyperedge_weighted_patched.csv
- cleaned case count: 1873
- case-label rows: 2119
- unmapped label rows: 0
- obsolete label rows: 64
- overlap_zero case-label rows: 777
- overlap_zero case rate（任一 gold label 有重叠即不算 zero）: 0.3449
- mean exact_overlap(mapped rows): 0.0939

## 判断

- 若 original 与 cleaned 的 HPO Jaccard 接近 1 且 lost_hpo 很少，则 overlap_zero 高不主要由 cleaned HPO 丢失导致。
- 若 `unmapped_label` 或 `obsolete_label` 很少，则 overlap_zero 不主要由 cleaned disease label 映射失败导致。
- 若 gold disease hyperedge 本身 HPO 数很少或为 0，则更像 disease-HPO knowledge base 覆盖不足。
- 对仍然 overlap_zero 的病例，需要回看原始 `text` 和 HPO 抽取证据，判断是否临床文本本身缺少典型表型或 HPO 抽取噪声。