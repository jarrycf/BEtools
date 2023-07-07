import numpy as np
import torch
from torch import nn

class SH_SelfAttention(nn.Module):
    """单头自注意力模块"""
    def __init__(self, input_size):
        
        super().__init__()
        # 定义查询、键和值的转换矩阵
        # 通常input_size等于embed_size
        self.embed_size = input_size
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.softmax = nn.Softmax(dim=2)  # 在特征维度上进行归一化
    
    def forward(self, X): # X: 200*20*64
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        X_q = self.Wq(X)  # 200*20*64
        X_k = self.Wk(X)  # 200*20*64
        X_v = self.Wv(X)  # 200*20*64
        
        # 将查询和键进行缩放
        X_q_scaled = X_q / (self.embed_size ** (1/4)) # 200*20*64
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2)) # 200*20*64 bmm 200*64*20 => 200*20*20
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w) # 200*20*20

        # 重新加权的值向量
        z = torch.bmm(attn_w_normalized, X_v) # 200*20*64
        # print(z.shape)
        
        return z, attn_w_normalized # z: 200*20*64  attn_w_normalized: 200*20*20
    

class MH_SelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, input_size, num_attn_heads): # 64 8
        
        super().__init__()
        layers = [SH_SelfAttention(input_size) for i in range(num_attn_heads)]
        self.multihead_pipeline = nn.ModuleList(layers)
        embed_size = input_size
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size)
        
    def forward(self, X): # 200*20*64
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        
        out = []
        bsize, num_positions, inp_dim = X.shape
        attn_tensor = X.new_zeros((bsize, num_positions, num_positions)) # 200*20*20
        for SH_layer in self.multihead_pipeline:
            z, attn_w = SH_layer(X) # z: 200*20*64  attn_w: 200*20*20
            out.append(z)
            attn_tensor += attn_w # 非全零了
        # 在特征维度上连接
        out = torch.cat(out, -1) # 200*20*(64*8) -> 200*20*512

        attn_tensor = attn_tensor/len(self.multihead_pipeline) # 200*20*20/8 => 200*20*20
        print()

        # 返回不同自注意力块的统一向量映射
        return self.Wz(out), attn_tensor # out: 200*20*64  attn_tensor: 200*20*20
        

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        
        super().__init__()
        
        embed_size = input_size
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads) #(64, 8)
        
        self.layernorm_1 = nn.LayerNorm(embed_size)

        # 也称为位置前馈神经网络
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X): # 200*20*64
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        # z是大小为(batch, sequence length, input_size)的张量
        z, attn_mhead_tensor = self.multihead_attn(X) # # out: 200*20*64  attn_mhead_tensor: 200*20*20
        # 层归一化与残差连接 
        z = self.layernorm_1(z + X) # 200*20*64
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)

        return z, attn_mhead_tensor # z: 200*20*64  attn_mhead_tensor: 200*20*20

"""
基于原始Transformer论文（'Attention is all what you need'）中提出的余弦和正弦方法实现位置编码器
"""
class NucleoPosEncoding(nn.Module):
    def __init__(self, num_nucleotides, seq_len, embed_dim, pdropout=0.1):
        super().__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embed_dim)
        self.dropout = nn.Dropout(p=pdropout)
        # 位置编码矩阵
        base_pow = 10000
        PE_matrix = torch.zeros((1, seq_len, embed_dim))
        i_num = torch.arange(0., seq_len).reshape(-1, 1)  # i在序列长度上迭代（即序列项）
        j_denom = torch.pow(base_pow, torch.arange(0., embed_dim, 2.) / embed_dim)  # j在嵌入维度上迭代
        PE_matrix[:, :, 0::2] = torch.sin(i_num/j_denom)
        PE_matrix[:, :, 1::2] = torch.cos(i_num/j_denom)
        self.register_buffer('PE_matrix', PE_matrix)
        
        
    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb(X)
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + self.PE_matrix
        return self.dropout(X_embpos)

class NucleoPosEmbedder(nn.Module):
    def __init__(self, num_nucleotides, seq_length, embedding_dim):
        super().__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)

    def forward(self, X): # X: 200*20
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb(X) # 200*20*64
        bsize, seqlen, featdim = X_emb.size() 
        device = X_emb.device
        positions = torch.arange(seqlen).to(device)
        positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim)
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + positions_emb # 200*20*64
        return X_embpos

class PerBaseFeatureEmbAttention(nn.Module):
    """逐个特征向量注意力模块"""
    def __init__(self, input_dim, seq_len): # 64 20
        
        super().__init__()
        # 定义查询、键和值的转换矩阵
        # 通常input_size等于embed_size
        self.embed_size = input_dim
        self.Q = nn.Parameter(torch.randn((seq_len, input_dim), dtype=torch.float32), requires_grad=True) # 20*64
        self.softmax = nn.Softmax(dim=-1)  # 在特征维度上进行归一化
    
    def forward(self, X): # 200*20*64
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        bsize, seqlen, featdim = X.shape
        X_q = self.Q[None, :, :].expand(bsize, seqlen, featdim)  # 20*64 => 200*20*64
        X_k = X #200*20*64
        X_v = X
        # 将查询和键进行缩放
        X_q_scaled = X_q / (self.embed_size ** (1/4)) # 200*20*64
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2)) # 200*20*20
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w) # 200*20*20
        
        z = torch.bmm(attn_w_normalized, X_v) # 200*20*64
        
        return z, attn_w_normalized # z: 200*20*64 attn_w_normalized: 200*20*20

class FeatureEmbAttention(nn.Module):
    def __init__(self, input_dim): # 64
        '''
        Args:
            input_dim: int, 输入向量的大小（即特征向量）
        '''
        super().__init__()
        self.input_dim = input_dim
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)  # 在序列长度上进行归一化

    def forward(self, X): # 200*20*64
        '''执行正向计算
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4)) # 200*20*64
        queryv_scaled = self.queryv / (self.input_dim ** (1/4)) # 200*20*64
        # 使用matmul来计算张量乘向量
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)  # 200*20

        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # 重新加权的值向量（在这种情况下，重新加权原始输入X）
        # 将attn_weights_norm展开为(bsize, 1, seqlen)
        # 与形状为(bsize, seqlen, feat_dim)的X进行批量乘法
        # 结果将为(bsize, 1, feat_dim)
        # 将结果压缩为(bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1) # 200*64
        
        # 返回(bsize, feat_dim), (bsize, seqlen)
        return z, attn_weights_norm # z: 200*64  attn_weights_norm: 200*20


class Categ_CrisCasTransformer(nn.Module):

    def __init__(self, input_size=64, num_nucleotides=4, 
                 seq_length=20, num_attn_heads=8, 
                 mlp_embed_factor=2, nonlin_func=nn.ReLU(), 
                 pdropout=0.3, num_transformer_units=12, 
                 pooling_mode='attn', num_classes=2, per_base=False): # per_base=True

        '''
        input_size=model_config.embed_dim, #64
        num_nucleotides=4, 
        seq_length=20, 
        num_attn_heads=model_config.num_attn_heads, # 8
        mlp_embed_factor=model_config.mlp_embed_factor, # 2
        nonlin_func=model_config.nonlin_func, 
        pdropout=model_config.p_dropout, # 0.1
        num_transformer_units=model_config.num_transformer_units, # 2
        pooling_mode='attn',
        num_classes=2)
        '''
        
        super().__init__()
        embed_size = input_size
        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size) # (4, 20, 64)    
        trfunit_layers = [TransformerUnit(input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout) 
                          for i in range(num_transformer_units)] # 2
        self.trfunit_pipeline = nn.ModuleList(trfunit_layers)
        self.per_base = per_base
        
        if not per_base: #per_base为false, 这不基于每个碱基进行处理
            self.pooling_mode = pooling_mode
            if pooling_mode == 'attn':
                self.pooling = FeatureEmbAttention(input_size) # 64
            elif pooling_mode == 'mean':
                self.pooling = torch.mean
            self.Wy = nn.Linear(embed_size, num_classes, bias=True)

        else:
            self.pooling_mode = pooling_mode
            if pooling_mode == 'attn':
                self.pooling = PerBaseFeatureEmbAttention(input_size, seq_length) # (64, 20)
            self.bias = nn.Parameter(torch.randn((seq_length, num_classes), dtype=torch.float32), requires_grad=True) # 20*2
            self.Wy = nn.Linear(embed_size, num_classes, bias=False) # 64 -> 2

        # 在特征维度上执行log softmax
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self._init_params_()
        
    def _init_params_(self):
        for p_name, p in self.named_parameters():
            param_dim = p.dim()
            if param_dim > 1:  # 权重矩阵
                nn.init.xavier_uniform_(p)
            elif param_dim == 1:  # 偏置参数
                if p_name.endswith('bias'):
                    nn.init.uniform_(p, a=-1.0, b=1.0)
                    # nn.init.xavier_uniform_(p)

    def forward(self, X): # 200*20
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        # (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(X) # 200*20*64
        bsize, num_positions, inp_dim = X_embpos.shape
        attn_tensor = X_embpos.new_zeros((bsize, num_positions, num_positions)) # 200*20*20
        xinput = X_embpos # 200*20*64
        for trfunit in self.trfunit_pipeline:
            z, attn_mhead_tensor = trfunit(xinput) # # z: 200*20*64  attn_mhead_tensor: 200*20*20
            xinput = z # 200*20*64
            attn_tensor += attn_mhead_tensor # 200*20*20
        attn_tensor = attn_tensor/len(self.trfunit_pipeline) # 200*20*20

         # 在seqlen向量上汇总
        if not self.per_base:
            if self.pooling_mode == 'attn':
                z, fattn_w_norm = self.pooling(z) # z: 200*64  attn_weights_norm: 200*20
            # 注意：z.mean(dim=1)或self.pooling(z, dim=1)会改变z的形状，变为(batch, embedding dim)
            # 我们可以通过运行z.mean(dim=1, keepdim=True)来保持维度，以得到(batch, 1, embedding dim)
            elif self.pooling_mode == 'mean':
                z = self.pooling(z, dim=1)
                fattn_w_norm = None
            y = self.Wy(z) # y: z: 200*2
        else:
            if self.pooling_mode == 'attn':
                z, fattn_w_norm = self.pooling(z) # z: 200*20*64 attn_w_normalized: 200*20*20
            y = self.Wy(z) + self.bias # 200*20*64 => 200*20*2 + 20*2 => 200*20*2
            # print(y.shape)
        
        return self.log_softmax(y), fattn_w_norm, attn_tensor # y: 200*20*2 fattn_w_norm: 200*20*20 attn_tensor: 200*20*20
