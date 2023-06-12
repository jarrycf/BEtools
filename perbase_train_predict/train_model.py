
import pandas as pd
from data_preprocess import * # generate_clean_df, generate_perbase_df
from dataset import *
from run_workflow import * # build_custom_config_map, test_run


# 一. 获取数据
base_editor = 'ABEmax'  # # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
if base_editor in {'ABEmax', 'ABE8e'}:
    target_base = 'A'
elif base_editor in {'BE4max', 'Target-AID'}:
    target_base = 'C'

data_file = "perbase_train_predict/data/M2_ESM.xlsx"
df = pd.read_excel(data_file, sheet_name=base_editor + "_perbase")


# 二. 数据预处理
# 1. 替换列名
column_mapping = {f'Position_{i}': f'V{i}' for i in range(1, 21)} 
column_mapping['Count'] = 'allCounts'
column_mapping['Sequence'] = 'seq'
df.rename(columns=column_mapping, inplace=True)
'''
        ID     Purpose               allCounts    seq                   V1  ...        V16  V17  V18  V19  V20
0      CTRL_g  Validation/Test       4704         ACTGAAGATCAGCATGTGTC  0.0 ...        0.0  0.0  0.0  0.0  0.0
'''

# 2. 处理目标值Y
df_clean = generate_clean_df(df)

# 3. 处理特征值X
df_perbase = generate_perbase_df(df_clean, target_base)
'''
eg: ACTGAAGATCAGCATGTGTC
M1 ... M20  ID      seq_type efficiency_score   edited_seq_categ    V1 ... V20  L1 ... L20  B1 ... B20
1  ... 0    CTRL_H  1        0.002860           1                   0  ... 0    A  ... C    0  ... 1
M: A会转成G的位置为 1
ID: 序列名称
seq_type: 被编辑 1
efficiency_score: 用于判断序列是否可被编辑
edited_seq_categ: 训练集 0
V: 每个位置的编辑效率得分 %
L: 每个位置的碱基
B: 被编码后的每一个碱基(['A', 'C', 'T', 'G'], [0,1,2,3])
'''

# 4. 创建dataset
criscas_datatensor = create_datatensor(df_perbase)
'''
return self.X_feat[indx], self.y_score[indx], self.y_categ[indx], indx, self.indx_seqid_map[indx]
'''

# 5. 数据集按分层抽样划分为5份训练集, 验证集, 测试集
data_partitions = get_stratified_partitions(df_perbase)
'''
{
    0: {'train': [2049, 2050, 2051, 2052,..], 
        'validation': [0, 4, 7, 9,..], 
        'test': [1, 2, 3, 5, 6, ...]},
    1: 
}
dict_keys([0, 1, 2, 3, 4])
'''

# 6. 判断每份训练集, 测试集, 验证集中的交集
validate_partitions(data_partitions)

# 7. 将data_partitions划分后的数据存入dataset并添加逆类别权重
data_partitions = generate_partition_datatensor(criscas_datatensor, data_partitions)
'''
{   
    0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
            target_id = self.partition_ids[indx]  0 -> 2049
            return self.criscas_datatensor[target_id]  return self.X_feat[indx], ..., indx, self.indx_seqid_map[indx]
        'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
        'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
        'class_weights': tensor([2.3655, 0.6340])},
    1: 
}
dict_keys([0, 1, 2, 3, 4])
'''

# 三. 模型训练
config_map = build_custom_config_map(experiment_desc="my_experiment", model_name="Transformer")
'''
(config, options)
config = {'dataloader_config': dataloader_config, # batch_size num_workers
            'model_config': hyperparam_config, # TrfHyperparamConfig(64, 8, 2, 0.1, nn.ReLU(), 2, 1e-3, 200, 20)
                        self.embed_dim = embed_dim # 64
                        self.num_attn_heads = num_attn_heads # 8
                        self.num_transformer_units = num_transformer_units # 2
                        self.p_dropout = p_dropout # 0.1
                        self.nonlin_func = nonlin_func # nn.ReLU()
                        self.mlp_embed_factor = mlp_embed_factor # 2
                        self.l2_reg = l2_reg # 0
                        self.batch_size = batch_size # 200
                        self.num_epochs = num_epochs # 20
            'generic_config': generic_config # fdtype to_gpu 
         }

options = {'experiment_desc': experiment_desc, # "my_experiment"
            'run_num': run_num, # -1
            'model_name': model_name, # "Transformer"
            'num_epochs': hyperparam_config.num_epochs, # 20
            'weight_decay': hyperparam_config.l2_reg} # 1e-3
          }
'''

train_val_dir = "{}".format(base_editor+'Transformerv1.0')
run_gpu_map = {run_num: 0 for run_num in data_partitions} # 运行编号和CPU索引的映射（使用同一个CPU索引）eg: {1: 0, 2: 0} 
num_epochs = 100

train_val_run(data_partitions, config_map, train_val_dir, run_gpu_map, num_epochs)

# 四. 模型评估
test_dir = "./{}".format(base_editor+'Transformerv1.0')
test_run(data_partitions, config_map, train_val_dir, test_dir, run_gpu_map, num_epochs)

