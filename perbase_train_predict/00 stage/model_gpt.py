import numpy as np
import torch
from torch import nn


class SH_SelfAttention(nn.Module):
    """单头自注意力模型，定义Query、Key和Value的变换矩阵。
    """
    def __init__(self, input_size):
        super(SH_SelfAttention, self).__init__()
        self.embed_size = input_size
        # 使用nn.Linear并设置bias=False可以减少参数数量，提高模型的运行速度和性能
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.scale = np.sqrt(self.embed_size)

    def forward(self, X):
        """前向传播函数，根据输入的X（batch, sequence length, input_size），
        计算并返回自注意力矩阵及其标准化的值。
        """
        X_q, X_k, X_v = self.Wq(X), self.Wk(X), self.Wv(X)
        attn_w = torch.bmm(X_q, X_k.transpose(1,2)) / self.scale
        attn_w_normalized = torch.softmax(attn_w, dim=-1)
        z = torch.bmm(attn_w_normalized, X_v)
        return z, attn_w_normalized


class MH_SelfAttention(nn.Module):
    """多头自注意力模型，通过堆叠多个单头自注意力模型来实现。
    """
    def __init__(self, input_size, num_attn_heads):
        super(MH_SelfAttention, self).__init__()
        self.multihead_pipeline = nn.ModuleList([SH_SelfAttention(input_size) for _ in range(num_attn_heads)])
        self.Wz = nn.Linear(num_attn_heads * input_size, input_size)

    def forward(self, X):
        """前向传播函数，对每个注意力头进行计算，然后将结果合并。
        """
        out, attn_dict = [], {}
        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(X)
            out.append(z)
            attn_dict[f'h{count}'] = attn_w
        out = torch.cat(out, -1)
        return self.Wz(out), attn_dict


class TransformerUnit(nn.Module):
    """Transformer单元，包括一个多头自注意力模型和一个前馈神经网络。
    """
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        super(TransformerUnit, self).__init__()
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads)
        self.layernorm_1 = nn.LayerNorm(input_size)
        self.MLP = nn.Sequential(
            nn.Linear(input_size, input_size * mlp_embed_factor),
            nonlin_func,
            nn.Linear(input_size * mlp_embed_factor, input_size)
        )
        self.layernorm_2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(pdropout)

    def forward(self, X):
        """前向传播函数，首先通过多头注意力模型，然后经过前馈神经网络，最后通过dropout和层标准化。
        """
        z, attn_mhead_dict = self.multihead_attn(X)
        z = self.dropout(self.layernorm_1(z + X))
        z = self.dropout(self.layernorm_2(self.MLP(z) + z))
        return z, attn_mhead_dict


class NucleoPosEmbedder(nn.Module):
    """嵌入模块，用于将核苷酸序列和位置信息编码为固定大小的向量。
    """
    def __init__(self, num_nucleotides, seq_length, embedding_dim):
        super(NucleoPosEmbedder, self).__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)

    def forward(self, X):
        """前向传播函数，将核苷酸序列和位置信息进行嵌入，然后将两者相加得到最终的嵌入向量。
        """
        X_emb = self.nucleo_emb(X)
        bsize, seqlen, featdim = X_emb.size()
        positions = torch.arange(seqlen, device=X.device)
        positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim)
        return X_emb + positions_emb


class FeatureEmbAttention(nn.Module):
    """特征嵌入注意力模型，定义对输入特征进行自注意力的机制。
    """
    def __init__(self, input_dim):
        super(FeatureEmbAttention, self).__init__()
        self.queryv = nn.Parameter(torch.randn(input_dim))
        self.scale = np.sqrt(input_dim)

    def forward(self, X):
        """前向传播函数，计算注意力权重并返回重新加权的输入特征向量。
        """
        attn_weights = X.matmul(self.queryv) / self.scale
        attn_weights_norm = torch.softmax(attn_weights, dim=1)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1)
        return z, attn_weights_norm


class Categ_CrisCasTransformer(nn.Module):
    """基于Transformer的分类模型，用于处理CrisCas数据。
    """
    def __init__(self, input_size=64, num_nucleotides=4, seq_length=20, num_attn_heads=8, 
                 mlp_embed_factor=2, nonlin_func=nn.ReLU(), pdropout=0.3, num_transformer_units=12, 
                 pooling_mode='attn', num_classes=2):
        super(Categ_CrisCasTransformer, self).__init__()
        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, input_size)
        self.trfunit_pipeline = nn.ModuleList([TransformerUnit(input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout) 
                                               for _ in range(num_transformer_units)])
        self.Wy = nn.Linear(input_size, num_classes)
        self.pooling = FeatureEmbAttention(input_size) if pooling_mode == 'attn' else torch.mean
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        """前
        向传播函数，首先对输入进行嵌入，然后通过Transformer单元进行处理，
        最后通过池化和全连接层进行分类。
        """
        X_embpos = self.nucleopos_embedder(X)
        attn_mlayer_mhead_dict = {}
        for count, trfunit in enumerate(self.trfunit_pipeline):
            X_embpos, attn_mhead_dict = trfunit(X_embpos)
            attn_mlayer_mhead_dict[f'l{count}'] = attn_mhead_dict

        if isinstance(self.pooling, nn.Module):
            z, fattn_w_norm = self.pooling(X_embpos)
        else:
            z = self.pooling(X_embpos, dim=1)
            fattn_w_norm = None
        y = self.log_softmax(self.Wy(z))
        return y, fattn_w_norm, attn_mlayer_mhead_dict
