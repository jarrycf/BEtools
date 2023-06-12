
import os
import pandas as pd

from perbase.utilities import * # create_directory, get_device, report_available_cuda_devices
from perbase.predict_model import *

# 设置环境
curr_pth = os.path.dirname(__file__) # /Users/jarrycf/Desktop/BEtools2.0/demo 
curr_pth = os.path.dirname(curr_pth) # /Users/jarrycf/Desktop/BEtools2.0

csv_dir = create_directory(os.path.join(curr_pth, "sample_data", "predictions")) #/Users/jarrycf/Desktop/BEtools2.0/sample_data/predictions
report_available_cuda_devices()
device = get_device(True, 0)


# 一. 获取数据
# teditor = input("输入基于编辑技术: ") # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
# teditor = 'ABEmax' # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
# teditor = 'ABEmax' # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
# teditor = 'ABE8e' # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
teditor = 'Target-AID' # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)


# 二. 数据预处理
seq_df = pd.read_csv(os.path.join(curr_pth, 'sample_data', 'abemax_sampledata.csv'), header=0) # 第一行作为列名（header）
'''
                  ID                   seq
0   CTRL_HEKsiteNO18  ACACACACACTTAGAATCTG
1       CTRL_RSF1NO2  ACCCATTAAAGTTGAGGTGA
'''

pos = 5
pred_option = 'mean' # 定义预测选项 mean, median, max


# 三. 模型预测
bedict = BEDICT_CriscasModel(teditor, device)
pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)
pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, pred_option)
# proc_df.to_csv(os.path.join(csv_dir, f'proc_df_allrun.csv'))
'''
	index	M1	M2	M3 ... M19	M20	ID	                seq	                    seq_type	L1	L2	L3 ... L19	L20	B1	B2	B3 ... B19	B20
0	0	    1	0	1      0	0	CTRL_HEKsiteNO18	ACACACACACTTAGAATCTG	1	        A	C	A  ... T	G	0	1	0  ... 2	3
1	1	    1	0	0      0	1	CTRL_RSF1NO2	    ACCCATTAAAGTTGAGGTGA	1	        A	C	C  ... G	A	0	1	1  ... 3	0
2	2	    1	0	1      0	0	CTRL_EMX1NO2	    AGATTTATGCAAACGGGTTG	1	        A	G	A  ... T	G	0	3	0  ... 2	3
'''
pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))
'''
	id	base_pos	prob_score_class0	prob_score_class1	Attn0	    Attn1          Attn19       run_id   model_name
0	CTRL_HEKsiteNO18	0	0.99941456	0.000585471	        0.045729	0.042664137    0.03622631   run_0    ABEmax
1	CTRL_HEKsiteNO18	2	0.97717017	0.022829875	        0.0482843	0.045546014    0.036456432  run_0    ABEmax
'''

pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{pred_option}.csv'))
'''
	id	base_pos	model_name	prob_score_class0	prob_score_class1	Attn0	      Attn1         Attn19
0	CTRL_DOCK3NO2	1	ABEmax	0.9959866	        0.004013466	        0.042520024	  0.052130606   0.042516273
1	CTRL_DOCK3NO2	2	ABEmax	0.98549235	        0.01450765	        0.042436536	  0.0532129     0.04271798
'''


# 四. 结果可视化
seqid_pos_map = {} # 用于存储序列 ID 和位置信息的映射
for i in range(len(seq_df)):
    # 获取seq_id和pos值
    seq_id = seq_df.iloc[i, 0]
    seqid_pos_map[seq_id] = [pos]
'''
{
    'CTRL_HEKsiteNO18': [5], 'CTRL_RSF1NO2': [5], 'CTRL_EMX1NO2': [5], 'CTRL_FANCFNO3': [5], 'CTRL_FANCFNO5': [5], 'CTRL_TARDBPNO2': [5], 'CTRL_HEKsiteNO1': [5], 'CTRL_HEKsiteNO9': [5], 'CTRL_HEKsiteNO2': [5], 'CTRL_EMX1NO1': [5], 'HEKsite7LV': [5], 'CTRL_HEKsiteNO11': [5], 'CTRL_HEKsiteNO14': [5], 'CTRL_HEKsiteNO8': [5], 'CTRL_HEKsiteNO3': [5], 'CTRL_DOCK3NO2': [5], 'CTRL_ZNF212NO1': [5], 'CTRL_NEK1NO2': [5]
}
'''

apply_attn_filter = False # 是否应用注意力分数过滤器
fig_dir =  create_directory(os.path.join(curr_pth, 'sample_data', 'fig_dir'))

bedict.highlight_attn_per_seq(pred_w_attn_runs_df, # 求的平均值的每个位置的表
                              proc_df, # 每个位置排列组合的表
                              seqid_pos_map=seqid_pos_map, # 位置和对应的ID
                              pred_option=pred_option, # 选择统计方法
                              apply_attnscore_filter=apply_attn_filter, # 不使用注意力过滤器
                              fig_dir=create_directory(os.path.join(fig_dir, pred_option))) # 文件路径

