
import os
import pandas as pd

from haplotype.dataset import *
from haplotype.data_preprocess import * # SeqProcessConfig, HaplotypeSeqProcessor, get_char, VizInpOutp_Haplotype, HaplotypeVizFile
from haplotype.utilities import * # create_directory, report_available_cuda_devices, get_device
from haplotype.predict_model import BEDICT_HaplotypeModel

# 设置环境
curr_pth = os.path.dirname(__file__) 
curr_pth = os.path.dirname(curr_pth) 
csv_dir = create_directory(os.path.join(curr_pth, "sample_data", "predictions_haplo"))
report_available_cuda_devices()
device = get_device(True, 0)


# 一. 获取数据
cnv_nucl_dict = {
    'ABEmax': ('A', 'G'),
    'ABE8e': ('A', 'G'),
    'BE4max': ('C', 'T'),
    'Target-AID': ('C', 'T')
}

teditor = input("输入基于的编辑技术: ") # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)
win_left, win_right = map(int, input("输入需要编辑的窗口范围(用空格隔开)：").split())
seqconfig_dataproc = SeqProcessConfig(20, (1,20), (win_left, win_right), 1)

if teditor in cnv_nucl_dict:
    cnv_nucl = cnv_nucl_dict[teditor]
    seq_processor = HaplotypeSeqProcessor(teditor, cnv_nucl, seqconfig_dataproc)
else:
    print("无法找到对应的编辑技术模型：", teditor)

df = pd.read_csv(os.path.join(curr_pth, 'sample_data', 'bystander_sampledata.csv'))
sample_df = df.loc[df['Editor'] == teditor].copy()


# 二. 数据预处理
seqconfig_datatensgen = SeqProcessConfig(20, (1,20), (1, 20), 1) 
bedict_haplo = BEDICT_HaplotypeModel(seq_processor, seqconfig_datatensgen, device)
dloader = bedict_haplo.prepare_data(sample_df,
                                    ['seq_id','Inp_seq'],
                                    outpseq_col=None,
                                    outcome_col=None,
                                    renormalize=False,
                                    batch_size=500)


# 三. 模型预测
num_runs = 5 # 设定模型运行次数
pred_df_lst = [] # 用于存储每次模型运行的预测结果
for run_num in range(num_runs):
    # 指定模型目录
    model_dir = os.path.join(curr_pth, 'trained_models', 'bystander', teditor, 'train_val', f'run_{run_num}')
    print('运行次数:', run_num)
    print('模型目录:', model_dir)

    pred_df = bedict_haplo.predict_from_dloader(dloader,
                                                model_dir, 
                                                outcome_col=None)
    pred_df['run_num'] = run_num # 添加模型运行次数的列
    pred_df_lst.append(pred_df) # 将当前模型运行的预测结果添加到列表中
    
pred_df_unif = pd.concat(pred_df_lst, axis=0, ignore_index=True) # 将所有模型运行的预测结果合并成一个数据框
check_na(pred_df_unif) # 检查是否存在缺失值
agg_df = bedict_haplo.compute_avg_predictions(pred_df_unif) # 计算所有模型运行的平均预测结果
check_na(agg_df) # 检查是否存在缺失值


# 四. 结果可视化
tseqids = sample_df['seq_id']
res_html = bedict_haplo.visualize_haplotype(agg_df, 
                                            tseqids, 
                                            ['seq_id','Inp_seq'], 
                                            'Outp_seq', 
                                            'pred_score', 
                                            predscore_thr=0.)


vf = HaplotypeVizFile(os.path.join(curr_pth, 'haplotype', 'viz_resources'))

# for seq_id in res_html:
#     vf.create(res_html[seq_id], csv_dir, f'{teditor}_{seq_id}_haplotype')


import webbrowser
for seq_id in res_html:
    output_file = f"{teditor}_{seq_id}_haplotype.html"
    vf.create(res_html[seq_id], csv_dir, f'{teditor}_{seq_id}_haplotype')
    filepath = os.path.join('sample_data', 'predictions_haplo', output_file)
    webbrowser.open(f"file://{os.path.abspath(filepath)}")


