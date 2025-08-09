import torch
import torch.nn as nn
import math
from rotary_embedding_torch import RotaryEmbedding
from dataclasses import dataclass

@dataclass
class EncoderOut:
    last_hidden_state:torch.Tensor
    cls_out:torch.Tensor
    # scope_logits: torch.Tensor


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,d_k):
        super(RotaryPositionalEmbedding,self).__init__()
        # dim 需要與每個 head 的 dim 相同
        self.rotary_emb = RotaryEmbedding(dim=d_k)
    
    def forward(self,x:torch.tensor):
        # x: query or key (batch,heads,seq_len,d_k) ， d_k = d_ebd // heads
        return self.rotary_emb.rotate_queries_or_keys(x)
    
class ConditionalLayerNorm(nn.Module):
    """
    此 Block 結合 Feature-wise linear modulation 以及 Conditional layer normalization
    """
    def __init__(self, d_model: int, num_scopes: int, hidden: int = 512, eps: float = 1e-5):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        # MLP 將類別 logits 轉為 scale & shift 參數
        self.mlp = nn.Sequential(
            nn.Linear(num_scopes, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2 * d_model)
        )

    def forward(self, h: torch.Tensor, cat_logits: torch.Tensor):
        # h: (B, L, d_model), cat_logits: (B, L, C)
        # 1. 計算 scope distribution 作為我的 condition
        p = torch.softmax(cat_logits, dim=-1)  # (B, L, C)
        # 2. 以 MLP 作為 FiLM generator (可以是其他 module) → gamma, beta 
        gamma_beta = self.mlp(p)               # (B, L, 2*d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1) # (B, L, d_model) ; # (B, L, d_model)
        # 3. LayerNorm 計算
        mean = h.mean(-1, keepdim=True)
        var = h.var(-1, keepdim=True, unbiased=False)
        h_norm = (h - mean) / torch.sqrt(var + self.eps)
        # 4. FiLM 調制
        return gamma * h_norm + beta
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_ebd:int,head_nums:int,dropout:float):
        super(MultiHeadAttentionBlock,self).__init__()
        self.d_ebd = d_ebd # Embedding vector size
        self.h = head_nums # Number of heads
        
        if d_ebd % head_nums != 0:
            raise ValueError(f'd_ebd is not divisible by h')

        self.d_k = d_ebd // head_nums # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_ebd,d_ebd,bias=False) # Wq
        self.w_k = nn.Linear(d_ebd,d_ebd,bias=False) # Wk
        self.w_v = nn.Linear(d_ebd,d_ebd,bias=False) # Wv
        self.w_o = nn.Linear(d_ebd,d_ebd,bias=False) # Wo
        self.rotary_positional_encoding = RotaryPositionalEmbedding(self.d_k)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h,seq_len,d_k) -> (batch,h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            # mask (batch,seq_len) -> (batch,1,1,seq_len) 
            mask = mask[:,None,None,:]
            # Write a very low value (indicating-inf) to the  positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e4)
        
        attention_scores = attention_scores.softmax(dim=-1) #(batch,h,seq_len,seq_len) # Apply softmax
        """attenton dropout"""
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch,h,seq_len,seq_len)  --> (batch,h,seq_len,d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value),attention_scores
    
    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (batch,seq_len,d_ebd) --> (batch,seq_len,d_ebd)
        key = self.w_k(k)   # (batch, seq_len, d_ebd) --> (batch, seq_len, d_ebd)
        value = self.w_v(v) # (batch, seq_len, d_ebd) --> (batch, seq_len, d_ebd)

        # (batch,seq_len,d_ebd) --> (batch,seq_len,h,d_k) --> (batch,h,seq_len,d_k)
        shapes = (query.shape[0],query.shape[1],self.h,self.d_k)
        query = query.view(*shapes).transpose(1,2)
        key = key.view(*shapes).transpose(1, 2)
        value = value.view(*shapes).transpose(1, 2)

        # (batch, h,seq_len,d_k) -> (batch,h,seq_len,d_k) ; 為每一層 Query、Key 使用 rotary_position_encoding
        query = self.rotary_positional_encoding(query)
        key = self.rotary_positional_encoding(key)
        
        # Calculate attention
        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_ebd)
        x = x.transpose(1,2).reshape(x.shape[0],-1,self.h*self.d_k)

        # Multiply by Wo (將所有 heads 的資訊進行一個特徵整合)
        # (batch, seq_len, d_ebd) --> (batch, seq_len, d_ebd)  
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self,d_ebd,dropout:float):
        super(ResidualConnection,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=d_ebd)
    
    """這裡採用 pre-norm 方式，也就是 layernorm 在 sublayer 之前，pytorch 官方的作法也是如此，原因是可讓梯度更穩定，表現更好"""
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))
    
class FeedForward_Layer(nn.Module):
    def __init__(self,d_ebd:int,d_ff:int,dropout:float):
        super(FeedForward_Layer,self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(d_ebd,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_ebd)
        )
        
    def forward(self,x):
        # x: (batch, seq_len, d_ebd) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_ebd)
        x = self.linears(x)
        return x

class Encoderblock(nn.Module):
    def __init__(
        self,
        d_ebd: int,
        nhead: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.self_atten = MultiHeadAttentionBlock(d_ebd, nhead, dropout)
        self.res0 = ResidualConnection(d_ebd, dropout)
        self.res1 = ResidualConnection(d_ebd, dropout)
        self.feedforward = FeedForward_Layer(d_ebd, d_ff, dropout)

            
    def forward(self, x, mask=None):
        # self-attention + residual
        x = self.res0(x, lambda y: self.self_atten(y, y, y, mask))
        x = self.res1(x, self.feedforward)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_ebd: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        number_of_layers: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            Encoderblock(d_ebd, n_head, d_ff, dropout)
            for _ in range(number_of_layers)
        ])
        self.norm = nn.LayerNorm(d_ebd)
        self.cls_embedding = nn.Parameter(torch.rand(1,1,d_ebd))

    def forward(self, x: torch.Tensor, mask=None) -> EncoderOut:
        
        batch_size = x.size(0)
        # [batch, 1, d_ebd]
        cls_token = self.cls_embedding.expand(batch_size, -1, -1)
        
        # # [batch, 1 + seq_len , d_ebd]
        x = torch.cat([cls_token, x], dim=1) 
        
        cls_mask = torch.ones((batch_size,1),dtype=mask.dtype, device=mask.device)
        #  [batch, seq_len] -> [batch, 1+seq_len]
        mask = torch.cat([mask,cls_mask],dim=1)
        
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return EncoderOut(
            # last_hidden_state = x
            last_hidden_state=x[:, 1:],
            cls_out=x[:, 0],
        )


# class Encoderblock(nn.Module):
#     def __init__(self,d_ebd:int,nhead:int,d_ff:int,dropout:float):
#         super(Encoderblock,self).__init__()
#         # self.self_atten = nn.MultiheadAttention(d_ebd,nhead,dropout,batch_first=True)
#         self.self_atten = MultiHeadAttentionBlock(d_ebd,nhead,dropout)
#         self.residual_connections = nn.ModuleList([ResidualConnection(d_ebd,dropout) for _ in range(2)])
#         self.feedforward = FeedForward_Layer(d_ebd,d_ff,dropout)

#     def forward(self,x,mask=None):
#         # x (batch,seq_len,d_ebd)
#         x = self.residual_connections[0](x,lambda x:self.self_atten(x,x,x,mask)) #  (batch,seq_len,d_ebd)
#         x = self.residual_connections[1](x,self.feedforward) #  (batch,seq_len,d_ebd)

#         return x
    



# class Encoderblock(nn.Module):
#     def __init__(
#         self,
#         d_ebd: int,
#         nhead: int,
#         d_ff: int,
#         dropout: float,
#         num_scopes: int,
#         cat_hidden: int = 512,
#         build_conlayer:bool=False
#     ):
#         super().__init__()
#         self.self_atten = MultiHeadAttentionBlock(d_ebd, nhead, dropout)
#         self.res0 = ResidualConnection(d_ebd, dropout)
#         self.res1 = ResidualConnection(d_ebd, dropout)
#         self.feedforward = FeedForward_Layer(d_ebd, d_ff, dropout)
#         # token classifier and FiLM
        
#         if build_conlayer:
#             self.proj = nn.Linear(d_ebd, num_scopes)
#             self.cond_ln = ConditionalLayerNorm(d_ebd, num_scopes, hidden=cat_hidden)
#         else:
#             self.cond_ln = None
            
#     def forward(self, x, mask=None):
#         # self-attention + residual
#         x = self.res0(x, lambda y: self.self_atten(y, y, y, mask))
#         # Separate CLS, token 
#         cls_tok = x[:, :1]  # (B,1,D)
#         tok_h = x[:, 1:]  # (B, L, D) ; 不包含 <s> token
#         # tok_h = x
        
#         if self.cond_ln:
#             # Token-level logits
#             logits = self.proj(tok_h)              # (B, L, C)
#             # FiLM on token features
#             cond_h = self.cond_ln(tok_h, logits)        # (B, L, D)
#         else:
#             logits = None
#             cond_h = tok_h
#         # Reassemble sequence: CLS + conditioned
#         x = torch.cat([cls_tok, cond_h], dim=1) # (B,L+1,D)

#         # x = cond_h
#         # feed-forward + residual
#         x = self.res1(x, self.feedforward)

#         return x

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         d_ebd: int,
#         n_head: int,
#         d_ff: int,
#         dropout: float,
#         number_of_layers: int,
#         num_scopes: int,
#         cat_hidden: int = 512,
#         use_conlayer:bool = False
#     ):
#         super().__init__()
#         if use_conlayer:
#             self.layers = nn.ModuleList([
#                 Encoderblock(d_ebd, n_head, d_ff, dropout, num_scopes, cat_hidden,
#                             build_conlayer = (i >= number_of_layers - 2)) # 最後兩層有 condition layer norm
#                 for i in range(number_of_layers)
#             ])
#         else:
#             self.layers = nn.ModuleList([
#                 Encoderblock(d_ebd, n_head, d_ff, dropout, num_scopes)
#                 for _ in range(number_of_layers)
#             ])
#         self.num_layers = number_of_layers
#         self.norm = nn.LayerNorm(d_ebd)
        
#         self.cls_embedding = nn.Parameter(torch.rand(1,1,d_ebd))

#     def forward(self, x: torch.Tensor, mask=None) -> EncoderOut:
        
#         batch_size = x.size(0)
#         # [batch, 1, d_ebd]
#         cls_token = self.cls_embedding.expand(batch_size, -1, -1)
        
#         # # [batch, 1 + seq_len , d_ebd]
#         x = torch.cat([cls_token, x], dim=1) 
        
#         cls_mask = torch.ones((batch_size,1),dtype=mask.dtype, device=mask.device)
#         #  [batch, seq_len] -> [batch, 1+seq_len]
#         mask = torch.cat([mask,cls_mask],dim=1)
        
#         for layer in self.layers:
#             x = layer(x, mask)

#         x = self.norm(x)
        
#         return EncoderOut(
#             # last_hidden_state = x
#             last_hidden_state=x[:, 1:],
#             cls_out=x[:, 0],
#         )