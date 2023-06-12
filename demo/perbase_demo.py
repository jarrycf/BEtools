
import os
import pandas as pd

from perbase.utilities import * # create_directory, get_device, report_available_cuda_devices
from perbase.predict_model import *

# 设置环境
curr_pth = os.path.dirname(__file__) 
curr_pth = os.path.dirname(curr_pth) 
csv_dir = create_directory(os.path.join(curr_pth, "sample_data", "predictions"))
report_available_cuda_devices()
device = get_device(True, 0)

# 一. 获取数据
teditor = input("输入基于编辑技术: ") # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)


# 二. 数据预处理
sample_df = input("输入要预测的序列: ") # ACACACACACTTAGAATCTG  ACAGAATTTGTTGAGGGCGA
data = {'ID': ['seq_0'],
        'seq': [sample_df]}
seq_df= pd.DataFrame(data)
pos = eval(input("输入你要预测编辑的序列位置(A或C所在位置,从1开始): ")) # 5
pred_option = input("输入预测的方式(mean, median, max): ")


# 三. 模型预测
bedict = BEDICT_CriscasModel(teditor, device)
pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)
pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, pred_option)

pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))
pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{pred_option}.csv'))


# 四. 结果可视化
seqid_pos_map = {'seq_0':[pos], 'seq_1':[pos]}

apply_attn_filter = False # 是否应用注意力分数过滤器
fig_dir =  create_directory(os.path.join(curr_pth, 'sample_data', 'fig_dir'))

bedict.highlight_attn_per_seq(pred_w_attn_runs_df, 
                              proc_df,
                              seqid_pos_map=seqid_pos_map,
                              pred_option=pred_option, 
                              apply_attnscore_filter=apply_attn_filter, 
                              fig_dir=create_directory(os.path.join(fig_dir, pred_option)))

# import webbrowser
# result_file = os.path.join(curr_pth, "sample_data", "fig_dir", pred_option, f"{teditor}_seqattn_seq_0_basepos_{pos}_predoption_{pred_option}.pdf")
# webbrowser.open(f"file://{os.path.abspath(result_file)}")