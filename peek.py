import pandas as pd
df = pd.read_excel(r'D:\RareDisease-traindata\LLLdataset\DiseaseHy\processed\DiseaseHyperedge_data_v3.xlsx', nrows=5)
with open('v3_info.txt', 'w') as f:
    f.write('Columns: ' + str(df.columns.tolist()) + '\n')
    f.write(df.to_string())
