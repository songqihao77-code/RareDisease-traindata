# DDD Near-Miss Audit

- gold 在 top50 且 rank>1 的样本数: 337。其中 top5 但非 top1: 148；top50 但 rank>5: 189。
- 平均 top1-gold HGNN score gap: 0.0904；中位 gold rank: 7.0。

## Ontology Relation Distribution
|relation|num_cases|ratio|
|---|---|---|
|shared_ancestor|183|0.5430|
|same_parent|102|0.3027|
|candidate_descendant_of_gold|21|0.0623|
|candidate_ancestor_of_gold|19|0.0564|
|unrelated_or_unknown|12|0.0356|

## Most Frequent Top1 Confusions
|gold_disease_id|gold_disease_name|predicted_disease_id|predicted_disease_name|confusion_count|average_gold_rank|average_score_gap|ontology_relation|
|---|---|---|---|---|---|---|---|
|MONDO:0010717|pyruvate dehydrogenase E1-alpha deficiency|MONDO:0018651|obsolete lipoyl transferase 2 deficiency|2|3.0000|0.0237|unrelated_or_unknown|
|MONDO:0020242|hereditary macular dystrophy|MONDO:0018146|macular telangiectasia type 1|2|3.0000|0.0349|shared_ancestor|
|MONDO:0005129|cataract|MONDO:0013067|cataract 34 multiple types|1|2.0000|0.0046|candidate_descendant_of_gold|
|MONDO:0007039|NF2-related schwannomatosis|MONDO:0014630|familial adenomatous polyposis 3|1|2.0000|0.0145|shared_ancestor|
|MONDO:0007585|exostoses, multiple, type 1|MONDO:0010846|exostoses, multiple, type III|1|2.0000|0.0023|same_parent|
|MONDO:0007630|North Carolina macular dystrophy|MONDO:0018146|macular telangiectasia type 1|1|2.0000|0.0164|shared_ancestor|
|MONDO:0007733|holoprosencephaly 3|MONDO:0007819|solitary median maxillary central incisor syndrome|1|2.0000|0.0339|candidate_descendant_of_gold|
|MONDO:0007986|metatropic dysplasia|MONDO:0008701|achondrogenesis type IA|1|2.0000|0.0045|shared_ancestor|
|MONDO:0008075|schwannomatosis|MONDO:0014299|LZTR1-related schwannomatosis|1|2.0000|0.0035|candidate_descendant_of_gold|
|MONDO:0008209|Char syndrome|MONDO:0014213|CTCF-related neurodevelopmental disorder|1|2.0000|0.0180|shared_ancestor|
|MONDO:0008244|piebaldism|MONDO:0013201|Waardenburg syndrome type 4B|1|2.0000|0.0068|shared_ancestor|
|MONDO:0008318|Proteus syndrome|MONDO:0013125|CLAPO syndrome|1|2.0000|0.0544|same_parent|
|MONDO:0008546|thanatophoric dysplasia type 1|MONDO:0008547|thanatophoric dysplasia type 2|1|2.0000|0.0095|same_parent|
|MONDO:0008612|tuberous sclerosis 1|MONDO:0013199|tuberous sclerosis 2|1|2.0000|0.0046|same_parent|
|MONDO:0008722|short chain acyl-CoA dehydrogenase deficiency|MONDO:0700250|mitochondrial complex IV deficiency, nuclear type 1|1|2.0000|0.0068|shared_ancestor|
|MONDO:0008767|neuronal ceroid lipofuscinosis 3|MONDO:0008769|neuronal ceroid lipofuscinosis 2|1|2.0000|0.0301|same_parent|
|MONDO:0008847|atrichia with papular lesions|MONDO:0007511|ectodermal dysplasia, trichoodontoonychial type|1|2.0000|0.0671|shared_ancestor|
|MONDO:0008861|3-methylcrotonyl-CoA carboxylase 1 deficiency|MONDO:0009475|isovaleric acidemia|1|2.0000|0.0042|shared_ancestor|
|MONDO:0008918|carnitine-acylcarnitine translocase deficiency|MONDO:0009282|multiple acyl-CoA dehydrogenase deficiency|1|2.0000|0.0280|shared_ancestor|
|MONDO:0009130|Dyggve-Melchior-Clausen disease|MONDO:0008477|spondylometaphyseal dysplasia, Kozlowski type|1|2.0000|0.0109|shared_ancestor|
|MONDO:0009353|homocystinuria due to methylene tetrahydrofolate reductase deficiency|MONDO:0009609|methylcobalamin deficiency type cblG|1|2.0000|0.0069|shared_ancestor|
|MONDO:0009561|alpha-mannosidosis|MONDO:0018149|GM1 gangliosidosis|1|2.0000|0.1121|same_parent|
|MONDO:0009603|3-hydroxyisobutyryl-CoA hydrolase deficiency|MONDO:0014314|sacral agenesis-abnormal ossification of the vertebral bodies-persistent notochordal canal syndrome|1|2.0000|0.1483|shared_ancestor|
|MONDO:0009728|nephronophthisis 1|MONDO:0013302|nephronophthisis 11|1|2.0000|0.0192|same_parent|
|MONDO:0009746|hereditary sensory and autonomic neuropathy type 4|MONDO:0012092|hereditary sensory and autonomic neuropathy type 5|1|2.0000|0.0005|same_parent|
|MONDO:0009968|renal tubular acidosis, distal, 2, with progressive sensorineural hearing loss|MONDO:0018440|autosomal recessive distal renal tubular acidosis|1|2.0000|0.0315|candidate_ancestor_of_gold|
|MONDO:0010083|succinic semialdehyde dehydrogenase deficiency|MONDO:0012960|intellectual disability, autosomal dominant 5|1|2.0000|0.0006|shared_ancestor|
|MONDO:0010183|methylmalonic aciduria and homocystinuria type cblF|MONDO:0010184|methylmalonic aciduria and homocystinuria type cblC|1|2.0000|0.0015|same_parent|
|MONDO:0010196|Werner syndrome|MONDO:0012590|XFE progeroid syndrome|1|2.0000|0.0058|same_parent|
|MONDO:0010211|xeroderma pigmentosum group C|MONDO:0010213|xeroderma pigmentosum group E|1|2.0000|0.0161|same_parent|

判断: 排序错误并不只来自随机噪声；same_parent/shared_ancestor/ancestor-descendant 类关系占有可观比例，说明 ontology-aware hard negative 与只在 top50 内的 evidence rerank 都有针对性。许多 pair 只出现一次，因此更适合按 relation/HPO-overlap 切片，而不是只记单个疾病对。