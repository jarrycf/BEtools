import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

class CrisCASDataTensor(Dataset):
    def __init__(self, X_feat, y_score, y_categ, indx_seqid_map):
        # B: batch elements; T: sequence length
        self.X_feat = X_feat  # 特征数据，tensor.float32, B x T, (将序列字符映射为0-3)
        self.y_score = y_score  # 分数数据，tensor.float32, B, (效率得分)
        self.y_categ = y_categ  # 类别数据，tensor.int64, B, (效率得分类别)
        self.indx_seqid_map = indx_seqid_map  # 索引到序列ID的映射
        self.num_samples = self.X_feat.size(0)  # 样本数量

    def __getitem__(self, indx):
        if self.y_score is None:
            return self.X_feat[indx], indx, self.indx_seqid_map[indx]
            
        return self.X_feat[indx], self.y_score[indx], self.y_categ[indx], indx, self.indx_seqid_map[indx]

    def __len__(self):
        return self.num_samples

class CrisCASSeqDataTensor(Dataset):
    def __init__(self, X_feat, y_score, y_categ, y_overall_categ, mask, indx_seqid_map):
        # B: batch elements; T: sequence length
        self.X_feat = X_feat  # 特征数据，tensor.float32, B x T, (将序列字符映射为0-3)
        self.y_score = y_score  # 分数数据，tensor.float32, B x T, (效率得分)
        self.y_categ = y_categ  # 类别数据，tensor.int64, B x T, (效率得分类别)
        self.y_overall_categ = y_overall_categ  # 总体类别数据，tensor.int64, B, (效率得分类别)
        self.mask = mask  # 掩码数据，tensor.int64, B x T, (指示目标碱基的布尔掩码)
        self.indx_seqid_map = indx_seqid_map  # 索引到序列ID的映射
        self.num_samples = self.X_feat.size(0)  # 样本数量 18

    def __getitem__(self, indx):
        if self.y_score is None:
            return self.X_feat[indx], self.mask[indx], indx, self.indx_seqid_map[indx]

        return self.X_feat[indx], self.y_score[indx], self.y_categ[indx], self.mask[indx], indx, self.indx_seqid_map[indx]

    def __len__(self):
        return self.num_samples

class PartitionDataTensor(Dataset):
    def __init__(self, criscas_datatensor, partition_ids, dsettype, run_num):
        self.criscas_datatensor = criscas_datatensor  # CrisCASDataTensor或CrisCasSeqDataTensor的实例
        self.partition_ids = partition_ids  # 序列索引的列表
        self.dsettype = dsettype  # 数据集类型（例如训练集、验证集、测试集）
        self.run_num = run_num  # 运行次数
        self.num_samples = len(self.partition_ids[:])  # 分区中的样本数量

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.criscas_datatensor[target_id]

    def __len__(self):
        return self.num_samples

def create_datatensor(df, per_base=False, refscore_available=True):
    """从处理/清理的数据框中创建DataTensor实例
    
    参数:
        df: pandas.DataFrame，经过generate_perbase_df处理的数据
        per_base=True: 为每个碱基（每个序列的每个元素）生成不同的分数和类别，而不是为整个序列生成一个总体的分数和类别 
        refscore_available: 是否提供参考分数（refscore）
    """
    
    # 索引 -> 序列ID的映射
    # X_tensor -> B x T（序列字符映射为0-3）
    # mask -> B x T（存在A或C字符的布尔掩码）
    # y -> B（效率得分或效率得分类别）
    if refscore_available:
        if not per_base:
            X_tensor = torch.from_numpy(df[[f'B{i}' for i in range(1, 21)]].values)
            y_score = torch.from_numpy(df['efficiency_score'].values)
            y_categ = torch.from_numpy(df['edited_seq_categ'].values)
            seqs_id = df['ID']
            indx_seqid_map = {i: seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASDataTensor(X_tensor, y_score, y_categ, indx_seqid_map)
        else:
            X_tensor = torch.from_numpy(df[[f'B{i}' for i in range(1, 21)]].values)
            y_score = torch.from_numpy(df[[f'ES{i}' for i in range(1, 21)]].values)
            y_categ = torch.from_numpy(df[[f'ECi{i}' for i in range(1, 21)]].astype(np.int64).values)
            y_overall_categ = torch.from_numpy(df['edited_seq_categ'].values)
            mask = torch.from_numpy(df[[f'M{i}' for i in range(1, 21)]].values)
            seqs_id = df['ID']
            indx_seqid_map = {i: seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASSeqDataTensor(X_tensor, y_score, y_categ, y_overall_categ, mask, indx_seqid_map)
    else:
        if not per_base:
            X_tensor = torch.from_numpy(df[[f'B{i}' for i in range(1, 21)]].values) # torch.from_numpy用于将数组转换成张量
            y_score = None
            y_categ = None
            seqs_id = df['ID']
            indx_seqid_map = {i: seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASDataTensor(X_tensor, y_score, y_categ, indx_seqid_map)
        else:
            X_tensor = torch.from_numpy(df[[f'B{i}' for i in range(1, 21)]].values)
            '''
        	B1	B19	B20
            0	2	3
            X_tensor: 输入特征 
            y_score: 目标标签
            y_categ: 每个序列的分类效率得分
            y_overall_categ: 整体的分类效率得分
            '''
            y_score = None
            y_categ = None
            y_overall_categ = None
            mask = torch.from_numpy(df[[f'M{i}' for i in range(1, 21)]].values)
            '''
            M1	M20		            
        0   1	0
            '''
            seqs_id = df['ID']
            indx_seqid_map = {i: seqs_id[i] for i in df.index.tolist()}
            '''
            {0: 'CTRL_HEKsiteNO18', 1: 'CTRL_RSF1NO2', 2: 'CTRL_EMX1NO2', 3: 'CTRL_FANCFNO3', 4: 'CTRL_FANCFNO5', 5: 'CTRL_TARDBPNO2', 6: 'CTRL_HEKsiteNO1', 7: 'CTRL_HEKsiteNO9', 8: 'CTRL_HEKsiteNO2', 9: 'CTRL_EMX1NO1', 10: 'HEKsite7LV', 11: 'CTRL_HEKsiteNO11', 12: 'CTRL_HEKsiteNO14', 13: 'CTRL_HEKsiteNO8', 14: 'CTRL_HEKsiteNO3', 15: 'CTRL_DOCK3NO2', 16: 'CTRL_ZNF212NO1', 17: 'CTRL_NEK1NO2'}
            '''

            dtensor = CrisCASSeqDataTensor(X_tensor, y_score, y_categ, y_overall_categ, mask, indx_seqid_map)

    return dtensor
