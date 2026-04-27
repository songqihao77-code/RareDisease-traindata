# DDD Top50 Miss Audit

- 样本数: 194/761 (25.49%)。
- gold 不在 disease index / candidate universe: 0。
- gold 无 disease-HPO hyperedge: 0；obsolete gold: 1。
- case 与 gold exact HPO overlap 为 0: 26/194。
- 平均 case HPO 数: 14.55；平均 gold disease HPO 数: 31.95；平均 IC overlap: 0.3020。

结论: top50 miss 不是由 disease index 大面积缺失造成；主要是 HGNN candidate recall 未覆盖与 case/gold HPO 证据弱或稀疏导致。部分样本存在 zero exact overlap，但多数 gold 仍在 disease hyperedge 中，说明后续应做 coverage/label audit，而不是直接改 encoder。

## Top Notes
|case_id|gold_disease_id|gold_disease_name|full_rank|case_hpo_count|gold_disease_hpo_count|case_gold_shared_hpo_count|case_gold_ic_weighted_overlap|top1_candidate_id|top1_relation_to_gold|mapping_notes|
|---|---|---|---|---|---|---|---|---|---|---|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_101|MONDO:0014575|Singleton-Merten syndrome 2|543|34|9|5|0.1376|MONDO:0008429|candidate_ancestor_of_gold|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_107|MONDO:0005129|cataract|186|1|29|0|0.0000|MONDO:0007289|candidate_descendant_of_gold|zero_exact_overlap;top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_109|MONDO:0033652|mitochondrial complex IV deficiency, nuclear type 17|203|36|11|4|0.0770|MONDO:0014856|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_111|MONDO:0008758|mitochondrial DNA depletion syndrome 4a|223|36|60|3|0.0658|MONDO:0013350|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_13|MONDO:0007888|hereditary leiomyomatosis and renal cell cancer|2809|4|11|0|0.0000|MONDO:0008231|shared_ancestor|zero_exact_overlap;top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_138|MONDO:0005129|cataract|1791|5|29|0|0.0000|MONDO:0009919|shared_ancestor|zero_exact_overlap;top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_142|MONDO:0032577|retinitis pigmentosa 83|57|3|10|2|1.0000|MONDO:0013402|same_parent|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_146|MONDO:0010827|retinitis pigmentosa 14|70|7|10|1|0.1222|MONDO:0019200|candidate_ancestor_of_gold|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_156|MONDO:0011579|late-onset retinal degeneration|280|11|23|2|0.2728|MONDO:0010557|same_parent|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_158|MONDO:0014195|microcornea-myopic chorioretinal atrophy|174|34|8|7|0.1789|MONDO:0800167|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_160|MONDO:0009875|achromatopsia 3|376|3|10|0|0.0000|MONDO:0014501|shared_ancestor|zero_exact_overlap;top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_161|MONDO:0009007|Jalili syndrome|74|21|9|4|0.1448|MONDO:0015993|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_163|MONDO:0013468|retinitis pigmentosa 59|74|5|17|1|0.1550|MONDO:0019200|candidate_ancestor_of_gold|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_170|MONDO:0004580|retinal degeneration|61|2|64|2|1.0000|MONDO:0019200|candidate_descendant_of_gold|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_18|MONDO:0007856|palmoplantar keratoderma-esophageal carcinoma syndrome|337|9|15|0|0.0000|MONDO:0007808|shared_ancestor|zero_exact_overlap;top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_182|MONDO:0009319|pantothenate kinase-associated neurodegeneration|514|2|60|1|0.4505|MONDO:0019200|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_184|MONDO:0013967|peroxisome biogenesis disorder 14B|852|4|19|2|0.5394|MONDO:0008341|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_185|MONDO:0008972|rhizomelic chondrodysplasia punctata type 1|822|3|28|1|0.2288|MONDO:0019200|shared_ancestor|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_194|MONDO:0011974|retinitis pigmentosa 7|137|2|8|1|0.4505|MONDO:0019200|candidate_ancestor_of_gold|top1_ontology_near_gold|
|test::LLLdataset/dataset/processed/test/DDD.csv::case_195|MONDO:0014093|retinitis pigmentosa 66|97|8|9|2|0.2206|MONDO:0014614|shared_ancestor|top1_ontology_near_gold|