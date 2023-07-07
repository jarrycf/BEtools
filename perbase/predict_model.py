
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
from .model import Categ_CrisCasTransformer
from .data_preprocess import process_perbase_df
from .dataset import create_datatensor
from .utilities import ReaderWriter, build_probscores_df, check_na
from .attnetion_analysis import filter_attn_rows
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']



class BEDICT_CriscasModel:
    def __init__(self, base_editor, device):
        self.base_editor = base_editor
        self.device = device

    def _process_df(self, df, target_base):
        """处理输入的数据帧"""
        print('--- 预处理输入的数据 ---')
        df = process_perbase_df(df, target_base) 
        return df

    def _construct_datatensor(self, df, refscore_available=False):
        """构建数据张量"""
        dtensor = create_datatensor(df, per_base=True, refscore_available=refscore_available)
        return dtensor

    def _construct_dloader(self, dtensor, batch_size):
        """构建数据加载器"""
        print('--- 创建数据张量 ---')
        dloader = DataLoader(dtensor,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            sampler=None)
        return dloader

    def _build_base_model(self):
        """构建基本模型"""

        print('--- 构建模型 ---')
        embed_dim = 64
        num_attn_heads = 8
        num_trf_units = 2
        pdropout = 0.1
        activ_func = nn.ReLU()
        mlp_embed_factor = 2
        num_classes = 2
        model = Categ_CrisCasTransformer(input_size=embed_dim, #64
                                        num_nucleotides=4, 
                                        seq_length=20, 
                                        num_attn_heads=num_attn_heads, # 8
                                        mlp_embed_factor=mlp_embed_factor, # 2
                                        nonlin_func=activ_func,  # relu
                                        pdropout=pdropout,  # 0.1
                                        num_transformer_units=num_trf_units, #2
                                        pooling_mode='attn',
                                        num_classes=num_classes, #2
                                        per_base=True)
        # model = Categ_CrisCasTransformer(input_size=embed_dim, #64
        #                         num_nucleotides=4, 
        #                         seq_length=20, 
        #                         num_attn_heads=num_attn_heads, # 8
        #                         mlp_embed_factor=mlp_embed_factor, # 2
        #                         nonlin_func=activ_func,  # relu
        #                         pdropout=pdropout,  # 0.1
        #                         num_transformer_units=num_trf_units, #2
        #                         pooling_mode='attn',
        #                         num_classes=num_classes, #2
        #                         )
        return model

    def _load_model_statedict_(self, model, run_num):
        """加载训练好的模型"""
        print('--- 加载训练好的模型 ---')
        base_dir = os.path.dirname(__file__) 
        curr_pth = os.path.dirname(base_dir) # /Users/jarrycf/Desktop/BEtools2.0
        run_pth = os.path.join(curr_pth, 'trained_models', 'perbase', self.base_editor, 'train_val', f'run_{run_num}') # /Users/jarrycf/Desktop/BEtools2.0/trained_models/perbase/ABEmax/train_val/run_3

        device = self.device

        model_name = 'Transformer'
        models = [(model, model_name)]

        # 加载状态字典路径
        state_dict_dir = os.path.join(run_pth, 'model_statedict') #..train_val/run_3/model_statedict
        for model, m_name in models:
            model.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

        # 更新模型的数据类型, 开启并指定设备
        for model, _ in models:
            model.type(torch.float32).to(device)
            model.eval() # 进入评估, 开始用模型进行预测

        return model

    def _run_prediction(self, model, dloader):
        """运行预测"""
        device = self.device
        prob_scores = [] # 存储预测的概率分数
        seqs_ids_lst = [] # 存储序列的ID
        base_pos_lst = [] # 存储目标基因位点的位置信息, 即每个样本目标基因位点的索引
        seqid_fattnw_map = {} # 存储序列ID与前层自注意力权重
        seqid_hattnw_map  = {} # 存储序列ID与后层自注意力权重

        for i_batch, samples_batch in enumerate(dloader):

            X_batch, mask, b_seqs_indx, b_seqs_id = samples_batch
            
            '''
            X_batch: 18*20
            # 四种碱基 "A", "T", "C", "G" 的编码。例如，"A" 对应于 0，"T" 对应于 1，"C" 对应于 2，"G" 对应于 3。
            tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 0, 3, 0, 0, 2, 1, 2, 3], # ACACACACACTTAGAATCTG
                    [0, 1, 1, 1, 0, 2, 2, 0, 0, 0, 3, 2, 2, 3, 0, 3, 3, 2, 3, 0],
                    [0, 3, 0, 2, 2, 2, 0, 2, 3, 1, 0, 0, 0, 1, 3, 3, 3, 2, 2, 3],
                    [0, 3, 1, 3, 3, 1, 3, 3, 1, 2, 3, 1, 0, 1, 0, 0, 1, 1, 0, 3],
                    [0, 3, 3, 1, 1, 1, 3, 3, 1, 3, 1, 0, 1, 3, 3, 2, 3, 3, 1, 3],
                    [1, 3, 3, 3, 0, 3, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 0, 3, 2, 0],
                    [3, 0, 0, 1, 0, 1, 0, 0, 0, 3, 1, 0, 2, 0, 3, 0, 1, 2, 3, 1],
                    [3, 0, 0, 3, 0, 1, 1, 0, 0, 3, 3, 0, 2, 0, 3, 0, 1, 2, 3, 1],
                    [3, 0, 3, 2, 0, 2, 3, 0, 3, 3, 1, 0, 2, 0, 3, 0, 1, 2, 3, 1],
                    [3, 0, 3, 2, 1, 1, 3, 0, 3, 1, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0],
                    [3, 0, 2, 0, 2, 2, 3, 0, 2, 0, 1, 0, 0, 0, 0, 2, 3, 3, 3, 3],
                    [3, 3, 0, 1, 0, 3, 3, 1, 0, 3, 1, 0, 2, 0, 3, 0, 1, 2, 3, 2],
                    [3, 3, 1, 2, 0, 0, 0, 3, 0, 1, 1, 0, 2, 0, 3, 0, 1, 2, 3, 2],
                    [3, 2, 0, 0, 0, 1, 0, 0, 0, 3, 1, 0, 2, 0, 3, 0, 1, 2, 3, 0],
                    [3, 2, 1, 0, 0, 3, 0, 0, 0, 3, 1, 0, 3, 0, 3, 0, 1, 2, 3, 1],
                    [2, 0, 0, 3, 0, 1, 2, 3, 0, 0, 1, 0, 0, 3, 0, 0, 2, 3, 3, 2],
                    [2, 3, 1, 0, 1, 1, 2, 3, 3, 1, 0, 2, 1, 0, 0, 1, 0, 1, 3, 3],
                    [2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 0, 2, 0, 3, 2, 0, 0]])
            注:
            会运行5次 
            mask: 标记哪些样本是有效的  0 代表无效
            tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                    [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]])

            b_seqs_indx: 存储了每个样本所属的序列索引 
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17])

            b_seqs_id: 表示批量样本的序列ID [CTRL_HEKsiteNO18, CTRL_RSF1NO2, CTRL_EMX1NO2,..]
            ('CTRL_HEKsiteNO18', 'CTRL_RSF1NO2', 'CTRL_EMX1NO2', 'CTRL_FANCFNO3', 'CTRL_FANCFNO5', 'CTRL_TARDBPNO2', 'CTRL_HEKsiteNO1', 'CTRL_HEKsiteNO9', 'CTRL_HEKsiteNO2', 'CTRL_EMX1NO1', 'HEKsite7LV', 'CTRL_HEKsiteNO11', 'CTRL_HEKsiteNO14', 'CTRL_HEKsiteNO8', 'CTRL_HEKsiteNO3', 'CTRL_DOCK3NO2', 'CTRL_ZNF212NO1', 'CTRL_NEK1NO2')

            '''

            X_batch = X_batch.to(device)
            mask = mask.to(device)

            with torch.set_grad_enabled(False): # 梯度计算为关闭状态。在这个上下文中，模型的参数将不会进行梯度计算和更新。

                logsoftmax_scores, fattn_w_scores, hattn_w_scores = model(X_batch)
                # print(hattn_w_scores)
                '''
                logsoftmax_scores: 每个类别的概率分数
                [   
                    [-1.0640e-03, -6.8463e+00], # 两个不同类别的概率 -0.001064 -6.8463
                    [-1.4199e-02, -4.2617e+00], # 由于logsoftmax 是 softmax 和对数函数的结合, 所以和不为1
                    [-4.8299e-02, -3.0544e+00],
                    [-2.9135e+00, -5.5817e-02],
                    [-3.0738e+00, -4.7348e-02],
                    [-5.5041e+00, -4.0783e-03],
                    [-1.1708e+00, -3.7124e-01],
                    [-4.6425e-01, -9.9050e-01],
                    [-8.0505e-02, -2.5594e+00],
                    [-4.6723e-01, -9.8547e-01],
                    [-2.8790e-02, -3.5621e+00],
                    [-1.8171e-02, -4.0170e+00],
                    [-7.3632e-04, -7.2142e+00],
                    [-3.0655e-03, -5.7891e+00],
                    [-1.5043e-04, -8.8021e+00],
                    [-2.9011e-04, -8.1453e+00],
                    [-4.0320e-04, -7.8163e+00],
                    [-6.8629e-04, -7.2845e+00],
                    [-1.9179e-04, -8.5591e+00],
                    [-2.7152e-04, -8.2118e+00]],

    
                fattn_w_scores: 前层自注意力权重
                [[
                [0.0279, 0.0860, 0.0255,  ..., 0.0749, 0.0299, 0.0509], # 前一层自注意力 对序列中每一个元素的注意力分配
                [0.0293, 0.0799, 0.0305,  ..., 0.0720, 0.0440, 0.0555],
                [0.0406, 0.0639, 0.0469,  ..., 0.0553, 0.0474, 0.0623],
                ...,

                hattn_w_scores: 后层自注意力权重
                tensor(
                [[
                [0.0416, 0.0274, 0.0699,  ..., 0.0306, 0.0371, 0.0424],
                [0.0467, 0.0334, 0.0719,  ..., 0.0364, 0.0426, 0.0448],
                [0.0437, 0.0277, 0.0694,  ..., 0.0312, 0.0368, 0.0409],
                ...,
                '''


                # 将前层自注意力权重（fattn_w_scores）与其对应的序列ID（b_seqs_id）以字典的形式更新到 seqid_fattnw_map 中 计算设备（如GPU）移动到CPU上
                seqid_fattnw_map.update({seqid:fattn_w_scores[c].detach().cpu() for c, seqid in enumerate(b_seqs_id)})
                seqid_hattnw_map.update({seqid:hattn_w_scores[c].detach().cpu() for c, seqid in enumerate(b_seqs_id)})

                # __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                # print('y_pred_clss.shape', y_pred_clss.shape)
                # use mask to retrieve relevant entries
                tindx= torch.where(mask.type(torch.bool)) #找到掩码张量（mask）中为真（True）的元素的索引，返回一个包含索引的元组（tuple）

                # pred_class.extend(y_pred_clss[tindx].view(-1).tolist())
                # 将模型输出的对数softmax分数（logsoftmax_scores）中对应于真实掩码的概率值经过指数化、移动到CPU，并转换为NumPy数组后
                prob_scores.append((torch.exp(logsoftmax_scores[tindx].detach().cpu())).numpy())
                
                # 将真实掩码对应的样本序列ID（b_seqs_id）中的元素，根据索引 tindx[0] 进行提取，并将提取的序列ID添加到 seqs_ids_lst 列表中。
                seqs_ids_lst.extend([b_seqs_id[i] for i in tindx[0].tolist()]) 
                # 将真实掩码对应的目标基因位点的索引（tindx[1]）转换为列表后，将列表中的元素逐个添加到 base_pos_lst 列表中，用于记录目标基因位点的位置信息。
                base_pos_lst.extend(tindx[1].tolist()) # positions of target base 


                # torch.cuda.ipc_collect()
                # torch.cuda.empty_cache()
        # end of epoch
        # print("+"*35)
        prob_scores_arr = np.concatenate(prob_scores, axis=0)
        predictions_df = build_probscores_df(seqs_ids_lst, prob_scores_arr, base_pos_lst)

        
        return seqid_fattnw_map, seqid_hattnw_map, predictions_df

        # dump attention weights
        # if wrk_dir:
        #     ReaderWriter.dump_data(seqid_fattnw_map, os.path.join(wrk_dir, 'seqid_fattnw_map.pkl'))
        #     predictions_df.to_csv(os.path.join(wrk_dir, f'predictions.csv'))
        

    def _join_prediction_w_attn(self, pred_df, seqid_fattnw_map, seqid_hattnw_map):
        """将预测结果与注意力权重连接"""
        attn_w_lst = []
        for seq_id in seqid_fattnw_map: 
            bpos = pred_df.loc[pred_df['id'] == seq_id,'base_pos'].values
            attn_w = seqid_fattnw_map[seq_id][bpos].numpy()
            hattn_w = seqid_hattnw_map[seq_id].numpy()
            upd_attn = np.matmul(attn_w, hattn_w)
            attn_w_lst.append(upd_attn)
            
        attn_w_df = pd.DataFrame(np.concatenate(attn_w_lst, axis=0))
        attn_w_df.columns = [f'Attn{i}' for i in range(20)]
        
        pred_w_attn_df = pd.concat([pred_df, attn_w_df], axis=1)
        check_na(pred_w_attn_df)
        
        return pred_w_attn_df


    def predict_from_dataframe(self, df, batch_size=500):
        # TODO: 为特定编辑器添加目标运行编号


        if self.base_editor in {'ABEmax', 'ABE8e'}:
            target_base = 'A'
        elif self.base_editor in {'BE4max', 'Target-AID'}:
            target_base = 'C'

        proc_df = self._process_df(df, target_base)
        '''
            M1	M20	ID	                seq	                seq_type    L1	L19	L20	B1	B19	B20
        0   1	0	CTRL_HEKsiteNO18	ACACACACACTTAGAATCTG	1	    A	T	G	0	2	3
        '''
        dtensor = self._construct_datatensor(proc_df)
        '''
        
        '''
        dloader = self._construct_dloader(dtensor, batch_size) # 500

        pred_w_attn_runs_df = pd.DataFrame()

        
        model = self._build_base_model()

        for run_num in range(5):
            self._load_model_statedict_(model, run_num)

            print(f'单碱基编辑器类型 {self.base_editor}  | 运行次数：{run_num}')
            seqid_fattnw_map, seqid_hattnw_map, pred_df = self._run_prediction(model, dloader)
            pred_w_attn_df = self._join_prediction_w_attn(pred_df, seqid_fattnw_map, seqid_hattnw_map)
            pred_w_attn_df['run_id'] = f'run_{run_num}'
            pred_w_attn_df['model_name'] = self.base_editor
            pred_w_attn_runs_df = pd.concat([pred_w_attn_runs_df, pred_w_attn_df])
        # reset index
        # 新设置DataFrame的索引 drop=True 原来的索引列将被丢弃 inplace=True原地修改DataFrame
        pred_w_attn_runs_df.reset_index(inplace=True, drop=True)

        return pred_w_attn_runs_df, proc_df
    
    def _filter_attn_rows(self, pred_w_attn_df, base_w):
        """过滤注意力权重行"""
        filtered_df = filter_attn_rows(pred_w_attn_df, base_w)
        return filtered_df

    def _select_prediction_run(self, gr_df, option):
        """选择预测结果"""
        gr_df['diff'] = (gr_df['prob_score_class1'] - gr_df['prob_score_class0']).abs()
        if option == 'median':
            choice = gr_df['diff'].median()
        elif option == 'max':
            choice = gr_df['diff'].max()
        elif option == 'min':
            choice = gr_df['diff'].min()
        cond = gr_df['diff'] == choice
        t_indx = np.where(cond)[0][0]

        return gr_df.iloc[t_indx]

    def select_prediction(self, pred_w_attn_runs_df, option):
        """选择预测结果"""
        assert option in {'mean', 'median', 'max', 'min'}, "selection option should be in {mean, median, min, max}!"
        if option == 'mean':
            pred_w_attn_df = pred_w_attn_runs_df.groupby(['id', 'base_pos', 'model_name']).mean().reset_index()
        else:
            pred_w_attn_df = pred_w_attn_runs_df.groupby(['id', 'base_pos', 'model_name']).apply(self._select_prediction_run, option).reset_index(drop=True)
        return pred_w_attn_df


    def _highlight_attn_scores(self, df, pred_option, model_name, cmap = 'YlOrRd', fig_dir=None):
        """高亮模型注意力分数"""
        # we index these axes from 0 subscript
        fig, ax = plt.subplots(figsize=(11,3), 
                               nrows=1, 
                               constrained_layout=True) 
        # 创建了一个新的图形，设置了图形的大小、行数，并启用了布局优化

        # 创建列表
        seq_id = df['id']
        attn_vars = [f'Attn{i}'for i in range(20)]
        letter_vars = [f'L{i}' for i in range(1,21)]
        prob = df['prob_score_class1'] # 从数据帧中提取prob_score_class1列并赋值给prob。
        base_pos = df['base_pos'] + 1 # 从数据帧中提取prob_score_class1列并赋值给prob。

        # attn_scores  = df[[f'Attn{i}'for i in range(20)]].values.astype(np.float).reshape(1,-1)
        # 从数据帧中提取20个注意力分数列，转换为浮点类型，然后转换成一个1行n列的二维数组
        attn_scores  = df[[f'Attn{i}'for i in range(20)]].values.astype(np.float64).reshape(1,-1)
        # 计算数据帧中20个注意力分数列的最大值。
        max_score = df[[f'Attn{i}'for i in range(20)]].max()
        # 从数据帧中提取20个字母列的值，然后转换成一个1行n列的二维列表。
        base_letters =  df[letter_vars].values.reshape(1,-1).tolist()
        # 定义颜色条的关键字参数，设置了颜色条的标签和方向。
        cbar_kws={'label': '注意力得分', 'orientation': 'horizontal'} # sns中色阶条方向: 水平方向
        # cbar_kws={'label': '注意力得分', 'orientation': 'vertical'} # sns中色阶条方向: 水平方向

    #     cmap='YlOrRd'
        # cmap = 'YlGnBu_r'
        # g = sns.heatmap(attn_scores, cmap=cmap,annot=base_letters,fmt="",linewidths=.5, cbar_kws=cbar_kws, ax=ax)        
        # ax.set_xticklabels(list(range(1,21))) # 设置x轴的刻度标签为1到20。
        # ax.set(xlabel='碱基位置', ylabel='') # 设置x轴和y轴的标签
        # ax.set_yticklabels(['']) # 设置y轴的刻度标签为空。
        # ax.text(20.4, 0.2 , '碱基所在的位置 = {}'.format(base_pos), bbox={'facecolor': 'orange', 'alpha': 0.2, 'edgecolor':'none', 'pad': 9},
        #     fontsize=12) # 在图上添加文字，表示基础位置.
        # ax.text(20.4, 0.65,r'该位置被编辑的可能性='+ '{:.2f}'.format(prob), bbox={'facecolor': 'magenta', 'alpha': 0.2, 'edgecolor':'none', 'pad': 8},
        #         fontsize=12) # 在图上添加文字，表示编辑概率。
        # ax.text(0.2, -0.2 ,r'序列名称={}'.format(seq_id), bbox={'facecolor': 'grey', 'alpha': 0.2, 'edgecolor':'none', 'pad': 10, },
        #             fontsize=12, ha='center') # 图上添加文字，表示序列id。
        
        # ax.tick_params(left=False,labelbottom=True) # 设置轴刻度参数，禁用左侧刻度并启用底部刻度标签。
        # if fig_dir: # 将图形保存为PDF文件，文件名由模型名称、序列id、基础位置和预测选项组成。
        #     fig.savefig(os.path.join(fig_dir,f'{model_name}_seqattn_{seq_id}_basepos_{base_pos}_predoption_{pred_option}.pdf'),bbox_inches='tight')
        #     plt.close() 
        # fig, ax = plt.subplots(figsize=(10, 10))  # 设置画布尺寸
        # plt.subplots_adjust(top=0.9)  # 调整顶部边距

        # cmap = 'YlGnBu_r'
        # cmap = 'Oranges'
        cmap = 'Blues'
        # g = sns.heatmap(attn_scores, cmap=cmap,annot=base_letters,fmt="",linewidths=.5, cbar_kws=cbar_kws, ax=ax)        
        # g = sns.heatmap(attn_scores, cmap=cmap,annot=base_letters,fmt="",linewidths=.5, cbar_kws=False, ax=ax)    
        sns.heatmap(attn_scores, cmap=cmap, annot=base_letters, fmt="", linewidths=1, cbar=False, ax=ax)    
        ax.set_xticklabels(list(range(1,21))) # 设置x轴的刻度标签为1到20。
        ax.set(xlabel='碱基位置', ylabel='') # 设置x轴和y轴的标签
        ax.set_yticklabels(['']) # 设置y轴的刻度标签为空。
        ax.text(13, -0.1 , '预测位置 = {}'.format(base_pos), bbox={ 'alpha': 0.2, 'pad': 3, },
            fontsize=12) # 在图上添加文字，表示基础位置.
        ax.text(17, -0.1,r'被编辑可能性='+ '{:.2f}'.format(prob), bbox={ 'alpha': 0.2, 'pad': 3, },
                fontsize=12) # 在图上添加文字，表示编辑概率。
        ax.text(5, -0.1 ,r'序列ID : {}'.format(seq_id), bbox={ 'alpha': 0.2, 'pad': 3, },
                    fontsize=12) # 图上添加文字，表示序列id。
        # ax.text(0.2, -0.1 ,r'编辑器类型 : {}'.format('ABE8e'), bbox={ 'alpha': 0.2, 'pad': 3, },fontsize=12) # 图上添加文字，表示序列id。
        ax.text(0.2, -0.1 ,r'编辑器类型 : {}'.format(model_name), bbox={ 'alpha': 0.2, 'pad': 3, },fontsize=12) # 图上添加文字，表示序列id。
        # 创建一个新的 axes 用于颜色条
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)

        # 在新的 axes 中添加颜色条
        cbar = plt.colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
        cbar.set_label('注意力得分')

        # if fig_dir: # 将图形保存为PDF文件，文件名由模型名称、序列id、基础位置和预测选项组成。
        #     fig.savefig(os.path.join(fig_dir,f'{model_name}_seqattn_{seq_id}_basepos_{base_pos}_predoption_{pred_option}.pdf'),bbox_inches='tight')
        #     plt.close() 
        if fig_dir: # 将图形保存为PDF文件，文件名由模型名称、序列id、基础位置和预测选项组成。
            fig.savefig(os.path.join(fig_dir,f'{model_name}_seqattn_{seq_id}_basepos_{base_pos}_predoption_{pred_option}.png'),bbox_inches='tight')
            plt.close() 



        return ax

    def highlight_attn_per_seq(self, pred_w_attn_runs_df, 
                               proc_df,
                               seqid_pos_map=None,
                               pred_option='mean', 
                               apply_attnscore_filter=False, 
                               fig_dir=None):
        """按序列高亮注意力"""
        letter_vars = [f'L{i}' for i in range(1,21)] # 列名 "L1"到"L20"
        if pred_option in {'mean', 'median', 'min', 'max'}:
            pred_w_attn_df = self.select_prediction(pred_w_attn_runs_df, pred_option)
        else:
            pred_w_attn_df = pred_w_attn_runs_df

        if apply_attnscore_filter:
            ''' 将注意力分数小于等于阈值的行过滤掉 '''
            base_w = 1.0/20
            pred_w_attn_df = self._filter_attn_rows(pred_w_attn_df, base_w)
            pred_w_attn_df.reset_index(inplace=True, drop=True)
            check_na(pred_w_attn_df)

        res_df = pd.merge(left=pred_w_attn_df,
                          right=proc_df[['ID'] + letter_vars],
                          how='left', 
                          left_on=['id'], 
                          right_on=['ID'])
        # 预测数据集(pred_w_attn_df)与处理数据集(proc_df)进行左连接，连接键为'id'和'ID'列，得到一个新的DataFrame(res_df)

        check_na(res_df)
    
        if seqid_pos_map:
            '''根据提供的seqid_pos_map逐个序列进行高亮处理；否则，根据序列ID和基因位置进行分组，并对每个组的DataFrame应用高亮处理。'''
            for seqid, t_pos in seqid_pos_map.items():
                print('seq_id:', seqid)
                if t_pos: # 如果提供了位置列表
                    # 减去1，因为基因位点的索引是从0到19
                    t_pos_upd = [bpos-1 for bpos in t_pos]
                    cond = (res_df['id'] == seqid) & (res_df['base_pos'].isin(t_pos_upd))
                    t_df = res_df.loc[cond].copy()
                    for rname, row in t_df.iterrows():
                        print(f"highlighting seqid:{row['id']}, pos:{row['base_pos']}") 
                        self._highlight_attn_scores(row, pred_option, self.base_editor, cmap='YlOrRd', fig_dir=fig_dir) 
                else:
                    cond = res_df['id'] == seqid
                    t_df = res_df.loc[cond]
                    for rname, row in t_df.iterrows():
                        print(f"highlighting seqid:{row['id']}, pos:{row['base_pos']+1}") 
                        self._highlight_attn_scores(row, pred_option, self.base_editor, cmap='YlOrRd', fig_dir=fig_dir) 
        else:
            for gr_name, gr_df in res_df.groupby(['id', 'base_pos']):
                print(f"highlighting seqid: {gr_name[0]}, pos: {gr_name[1]+1}")
                for rname, row in gr_df.iterrows():
                    self._highlight_attn_scores(row, pred_option, self.base_editor, cmap='YlOrRd', fig_dir=fig_dir)

