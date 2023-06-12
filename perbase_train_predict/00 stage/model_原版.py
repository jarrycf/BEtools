import numpy as np
import torch
from torch import nn

class SH_SelfAttention(nn.Module):
    '''
    单头自注意力模型
        定义Query、Key和Value的变换矩阵
        通常 input_size 等于 embed_size
        使用nn.Linear并设置bias=False可以减少参数数量，提高模型的运行速度和性能
    ''' 

    def __init__(self, input_size):
        super(SH_SelfAttention, self).__init__() 
        self.embed_size = input_size # 64
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False) # 64 -> 64
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False) # 64 -> 64
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False) # 
        self.softmax = nn.Softmax(dim=2) # normalized across feature dimension
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        X_q = self.Wq(X) # queries 2*20*64
        X_k = self.Wk(X) # keys 2*20*64
        X_v = self.Wv(X) # values 2*20*64
        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4)) # 2*20*64 除以的是每个数
        X_k_scaled = X_k / (self.embed_size ** (1/4)) # 2*20*64
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2)) # 2*20*64 dot 2*64*20 => 2*20*20
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w) # 在第三个维度进行softmax # 2*20*20

        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v) # 2*20*20 dot 2*20*64 => 2*20*64
        
        return z, attn_w_normalized
    

class MH_SelfAttention(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads): # 64 4
        
        super(MH_SelfAttention, self).__init__()
        
        layers = [SH_SelfAttention(input_size) for i in range(num_attn_heads)] # 4
        
        self.multihead_pipeline = nn.ModuleList(layers)  # 4
        embed_size = input_size
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size) 
        
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
            2*20*64
        """
        
        out = []
        attn_dict = {}
        for count, SH_layer in enumerate(self.multihead_pipeline): # 4
            z, attn_w = SH_layer(X) # 2*20*64 2*20*20
            out.append(z)
            attn_dict[f'h{count}'] = attn_w # {h0: 2*20*20, h1: 2*20*20 }
        # concat on the feature dimension
        out = torch.cat(out, -1) # (2, 20, 4*64)
        
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out), attn_dict # 2*20*64  {h0: 2*20*20, h1: 2*20*20}
        

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        '''
        input_size: 64
        num_attn_heads: 4
        mlp_embed_factor: 2
        nonlin_func: nn.Relu()
        pdropout: 0.3
        '''
        super(TransformerUnit, self).__init__()
        embed_size = input_size # 64
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads) # (64 4) 
        self.layernorm_1 = nn.LayerNorm(embed_size)
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor), # 64 -> 128
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size) # 128 -> 54
        )
        self.layernorm_2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        # z is tensor of size (batch, sequence length, input_size)
        z, attn_mhead_dict = self.multihead_attn(X) # 2*20*64  {h0: 2*20*20, h1: 2*20*@0 }
        # layer norm with residual connection
        z = self.layernorm_1(z + X) # 2*20*64
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z) # 2*20*64
        
        return z, attn_mhead_dict # 2*20*64   {h0: 2*20*20, h1: 2*20*@0 }

"""
TODO: implement position encoder based on cosine and sine approach proposed 
      by original Transformers paper ('Attention is all what you need')
"""
        
class NucleoPosEmbedder(nn.Module):
    def __init__(self, num_nucleotides, seq_length, embedding_dim): # (4, 20, 64)
        super(NucleoPosEmbedder, self).__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim) # (4, 64)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim) # (20, 64)

    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
            假设 2*20
        """
        X_emb = self.nucleo_emb(X) # 200*20*64
        bsize, seqlen, featdim = X_emb.size() # 200*20*64
        device = X_emb.device 
        positions = torch.arange(seqlen).to(device)
        positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim) # 2*20*64
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + positions_emb # 200*20*64

        return X_embpos # 200*20*64

class FeatureEmbAttention(nn.Module):
    def __init__(self, input_dim):
        '''
        Args:
            attn_method: string, {'additive', 'dot', 'dot_scaled'}
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super(FeatureEmbAttention, self).__init__()
        self.input_dim = input_dim
        # use this as query vector against the transformer outputs
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True) # 1*64
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X): # 2*20*64
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4)) # 2*20*64
        queryv_scaled = self.queryv / (self.input_dim ** (1/4)) # 1*64
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled) # 2*20

        # softmax
        attn_weights_norm = self.softmax(attn_weights) # 2*20

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1) # 2*1*20 bmm 2*20*64 = 2*20*64 => 2*64

        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_weights_norm #  2*64 2*20
    

class Categ_CrisCasTransformer(nn.Module):

    def __init__(self, input_size=64, num_nucleotides=4, 
                 seq_length=20, num_attn_heads=8, 
                 mlp_embed_factor=2, nonlin_func=nn.ReLU(), 
                 pdropout=0.3, num_transformer_units=12, 
                 pooling_mode='attn', num_classes=2):
        
        super(Categ_CrisCasTransformer, self).__init__()
        
        embed_size = input_size

        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size) # (4, 20, 64) 核苷酸的种类数量 4
        
        trfunit_layers = [TransformerUnit(input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout) 
                          for i in range(num_transformer_units)]
        # self.trfunit_layers = trfunit_layers
        self.trfunit_pipeline = nn.ModuleList(trfunit_layers) # 12
        # self.trfunit_pipeline = nn.Sequential(*trfunit_layers)

        self.Wy = nn.Linear(embed_size, num_classes) # 64 => 4
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attn':
            self.pooling = FeatureEmbAttention(input_size)
        elif pooling_mode == 'mean':
            self.pooling = torch.mean
        # perform log softmax on the feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    
    def forward(self, X): # 200*20
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
            输入是200*20的数据, 这边假设2*20进行推演
        """
        # (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(X) # 200*20*64
        # z is tensor of size (batch,  seqlen, embedding dim)
        # z = self.trfunit_pipeline(X_embpos)
        attn_mlayer_mhead_dict = {}
        xinput = X_embpos
        for count, trfunit in enumerate(self.trfunit_pipeline):
            z, attn_mhead_dict = trfunit(xinput) # 2*20*64 => 2*20*64  {h0: 2*20*20, h1: 2*20*@0 }
            attn_mlayer_mhead_dict[f'l{count}'] = attn_mhead_dict # {l1: {h0: 2*20*20, h1: 2*20*@0 }, l2:}
            xinput = z # 2*20*64

         # pool across seqlen vectors

        if self.pooling_mode == 'attn':
            z, fattn_w_norm = self.pooling(z) # 2*20 2*64
        # Note: z.mean(dim=1) or self.pooling(z, dim=1) will change shape of z to become (batch, embedding dim)
        # we can keep dimension by running z.mean(dim=1, keepdim=True) to have (batch, 1, embedding dim)
        elif self.pooling_mode == 'mean':
            z = self.pooling(z, dim=1) 
            fattn_w_norm = None
        y = self.Wy(z) # 2*4
        
        return self.log_softmax(y), fattn_w_norm, attn_mlayer_mhead_dict # 2*4 2*64 {l1: {h0: 2*20*20, h1: 2*20*@0 }, l2:}