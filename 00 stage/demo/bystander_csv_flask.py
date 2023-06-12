from flask import Flask, render_template, request
import os
import pandas as pd
from haplotype.dataset import *
from haplotype.data_preprocess import *
from haplotype.utilities import *
from haplotype.predict_model import BEDICT_HaplotypeModel

app = Flask(__name__, template_folder="./templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 获取用户上传的 CSV 文件
    csv_file = request.files["csv_file"]
    if not csv_file:
        return "未上传 CSV 文件"

    # 获取用户输入
    teditor = request.form["teditor"]
    win_left = int(request.form["win_left"])
    win_right = int(request.form["win_right"])

    # 将 CSV 文件读入 DataFrame
    df = pd.read_csv(csv_file)

    # 运行原始代码
    curr_pth = os.path.dirname(__file__) 
    curr_pth = os.path.dirname(curr_pth) 
    csv_dir = create_directory(os.path.join(curr_pth, "sample_data", "predictions_haplo"))
    report_available_cuda_devices()
    device = get_device(True, 0)

    cnv_nucl_dict = {
        'ABEmax': ('A', 'G'),
        'ABE8e': ('A', 'G'),
        'BE4max': ('C', 'T'),
        'Target-AID': ('C', 'T')
    }

    seqconfig_dataproc = SeqProcessConfig(20, (1,20), (win_left, win_right), 1)

    if teditor in cnv_nucl_dict:
        cnv_nucl = cnv_nucl_dict[teditor]
        seq_processor = HaplotypeSeqProcessor(teditor, cnv_nucl, seqconfig_dataproc)
    else:
        print("无法找到对应的编辑技术模型：", teditor)

    sample_df = df.loc[df['Editor'] == teditor].copy()

    seqconfig_datatensgen = SeqProcessConfig(20, (1,20), (1, 20), 1) 
    bedict_haplo = BEDICT_HaplotypeModel(seq_processor, seqconfig_datatensgen, device)
    dloader = bedict_haplo.prepare_data(sample_df,
                                        ['seq_id','Inp_seq'],
                                        outpseq_col=None,
                                        outcome_col=None,
                                        renormalize=False,
                                        batch_size=500)

    num_runs = 5
    pred_df_lst = []
    for run_num in range(num_runs):
        model_dir = os.path.join(curr_pth, 'trained_models', 'bystander', teditor, 'train_val', f'run_{run_num}')
        print('运行次数:', run_num)
        print('模型目录:', model_dir)

        pred_df = bedict_haplo.predict_from_dloader(dloader,
                                                    model_dir, 
                                                    outcome_col=None)
        pred_df['run_num'] = run_num
        pred_df_lst.append(pred_df)
    
    pred_df_unif = pd.concat(pred_df_lst, axis=0, ignore_index=True)
    check_na(pred_df_unif)
    agg_df = bedict_haplo.compute_avg_predictions(pred_df_unif)
    check_na(agg_df)

    tseqids = sample_df['seq_id']
    res_html = bedict_haplo.visualize_haplotype(agg_df, 
                                                tseqids, 
                                                ['seq_id','Inp_seq'], 
                                                'Outp_seq', 
                                                'pred_score', 
                                                predscore_thr=0.)

    vf = HaplotypeVizFile(os.path.join(curr_pth, 'haplotype', 'viz_resources'))

    # 保存 HTML 文件
    html_pages = []
    for seq_id in res_html:
        output_file = f"{teditor}_{seq_id}_haplotype.html"
        vf.create(res_html[seq_id], '../sample_data/predictions_haplo', f'{teditor}_{seq_id}_haplotype')
        filepath = os.path.join('sample_data', 'predictions_haplo', output_file)
        html_pages.append(filepath)

    # 返回所有 HTML 页面
    # return render_template("index.html", html_pages=html_pages)
    html_pages = []
    for seq_id in res_html:
        html_pages.append(render_template(f"{teditor}_{seq_id}_haplotype.html"))
    return "\n".join(html_pages) 


if __name__ == "__main__":
    app.run()