import os
import sys
import pandas as pd


sys.path.append('../')
from perbase.utilities import * # create_directory, get_device, report_available_cuda_devices
from perbase.predict_model import *


from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 设置环境
        curr_pth = os.path.dirname(__file__)
        curr_pth = os.path.dirname(curr_pth)
        csv_dir = create_directory(os.path.join(curr_pth, "sample_data", "predictions"))
        report_available_cuda_devices()
        device = get_device(True, 0)

        # 一. 获取数据
        teditor = request.form['teditor'] # ABEmax(A->G) ABE8e(A->G) BE4max(C->T) Target-AID(C-T)

        # 二. 数据预处理
        sample_df = request.form['sample_df'] # ACACACACACTTAGAATCTG  ACAGAATTTGTTGAGGGCGA
        data = {'ID': ['seq_0'],
                'seq': [sample_df]}
        seq_df = pd.DataFrame(data)
        pos = int(request.form['pos']) # 5
        pred_option = request.form['pred_option'] # mean, median, max

        
        # 三. 模型预测
        bedict = BEDICT_CriscasModel(teditor, device)
        pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)
        pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, pred_option)

        pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))
        pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{pred_option}.csv'))

        # 四. 结果可视化
        seqid_pos_map = {'seq_0': [pos], 'seq_1': [pos]}

        apply_attn_filter = False
        fig_dir = create_directory(os.path.join(curr_pth, 'sample_data', 'fig_dir'))

        bedict.highlight_attn_per_seq(pred_w_attn_runs_df,
                                      proc_df,
                                      seqid_pos_map=seqid_pos_map,
                                      pred_option=pred_option,
                                      apply_attnscore_filter=apply_attn_filter,
                                      fig_dir=create_directory(os.path.join(fig_dir, pred_option)))

        # 生成的PDF文件路径
        # pdf_path = os.path.join(fig_dir, pred_option, f'ABE8e_seqattn_seq_0_basepos_{pos}_predoption_{pred_option}.png')

        png_path = os.path.join(fig_dir,'mean', f'{teditor}_seqattn_seq_0_basepos_{pos}_predoption_mean.png')
        print(png_path)
        
        with open(png_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode("ascii")

   

        return f"<img src='data:image/png;base64,{data}' style='max-width: 800px;'/>"

    return render_template('hello.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
