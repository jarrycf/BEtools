import numpy as np
import torch
from torch import nn

class SH_SelfAttention(nn.Module):
    """ single head self-attention module
    """
    def __init__(self, input_size):
        
        super(SH_SelfAttention, self).__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = input_size
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.softmax = nn.Softmax(dim=2) # normalized across feature dimension
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        X_q = self.Wq(X) # queries
        X_k = self.Wk(X) # keys
        X_v = self.Wv(X) # values
        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        
        return z, attn_w_normalized
    

class MH_SelfAttention(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super(MH_SelfAttention, self).__init__()
        
        layers = [SH_SelfAttention(input_size) for i in range(num_attn_heads)]
        
        self.multihead_pipeline = nn.ModuleList(layers)
        embed_size = input_size
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size)
        
    
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        
        out = []
        attn_dict = {}
        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(X)
            out.append(z)
            attn_dict[f'h{count}'] = attn_w
        # concat on the feature dimension
        out = torch.cat(out, -1) 
        
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out), attn_dict
        

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        
        super(TransformerUnit, self).__init__()
        
        embed_size = input_size
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads)
        
        self.layernorm_1 = nn.LayerNorm(embed_size)
        
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        # z is tensor of size (batch, sequence length, input_size)
        z, attn_mhead_dict = self.multihead_attn(X)
        # layer norm with residual connection
        z = self.layernorm_1(z + X)
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)
        
        return z, attn_mhead_dict

"""
TODO: implement position encoder based on cosine and sine approach proposed 
      by original Transformers paper ('Attention is all what you need')
"""
        
class NucleoPosEmbedder(nn.Module):
    def __init__(self, num_nucleotides, seq_length, embedding_dim):
        super(NucleoPosEmbedder, self).__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)

    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb(X)
        bsize, seqlen, featdim = X_emb.size()
        device = X_emb.device
        positions = torch.arange(seqlen).to(device)
        positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim)
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + positions_emb
        return X_embpos

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
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)

        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_weights_norm

class Categ_CrisCasTransformer(nn.Module):

    def __init__(self, input_size=64, num_nucleotides=4, 
                 seq_length=20, num_attn_heads=8, 
                 mlp_embed_factor=2, nonlin_func=nn.ReLU(), 
                 pdropout=0.3, num_transformer_units=12, 
                 pooling_mode='attn', num_classes=2):
        
        super(Categ_CrisCasTransformer, self).__init__()
        
        embed_size = input_size

        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size)
        
        trfunit_layers = [TransformerUnit(input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout) 
                          for i in range(num_transformer_units)]
        # self.trfunit_layers = trfunit_layers
        self.trfunit_pipeline = nn.ModuleList(trfunit_layers)
        # self.trfunit_pipeline = nn.Sequential(*trfunit_layers)

        self.Wy = nn.Linear(embed_size, num_classes)
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attn':
            self.pooling = FeatureEmbAttention(input_size)
        elif pooling_mode == 'mean':
            self.pooling = torch.mean
        # perform log softmax on the feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    
    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        # (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(X)
        # z is tensor of size (batch,  seqlen, embedding dim)
        # z = self.trfunit_pipeline(X_embpos)
        attn_mlayer_mhead_dict = {}
        xinput = X_embpos
        for count, trfunit in enumerate(self.trfunit_pipeline):
            z, attn_mhead_dict = trfunit(xinput)
            attn_mlayer_mhead_dict[f'l{count}'] = attn_mhead_dict
            xinput = z

         # pool across seqlen vectors

        if self.pooling_mode == 'attn':
            z, fattn_w_norm = self.pooling(z)
        # Note: z.mean(dim=1) or self.pooling(z, dim=1) will change shape of z to become (batch, embedding dim)
        # we can keep dimension by running z.mean(dim=1, keepdim=True) to have (batch, 1, embedding dim)
        elif self.pooling_mode == 'mean':
            z = self.pooling(z, dim=1)
            fattn_w_norm = None
        y = self.Wy(z) 
        
        return self.log_softmax(y), fattn_w_norm, attn_mlayer_mhead_dict