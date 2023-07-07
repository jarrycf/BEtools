
import pandas as pd
from data_preprocess import * # generate_clean_df, generate_perbase_df
from dataset import *
from run_workflow import * # build_custom_config_map, test_run


base_editor = 'ABEmax'  # # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
if base_editor in {'ABEmax', 'ABE8e'}:
    target_base = 'A'
elif base_editor in {'BE4max', 'Target-AID'}:
    target_base = 'C'

# data_file = "perbase_train/data/M2_ESM.xlsx"
data_file = "./data/M2_ESM.xlsx"
df = pd.read_excel(data_file, sheet_name=base_editor + "_perbase")

column_mapping = {f'Position_{i}': f'V{i}' for i in range(1, 21)} 
column_mapping['Count'] = 'allCounts'
column_mapping['Sequence'] = 'seq'
df.rename(columns=column_mapping, inplace=True)

editbase_cols = [f'V{i}' for i in range(1,21)]
y_score = df[editbase_cols].values # 0会被编辑
print(y_score.shape) # 10787*20