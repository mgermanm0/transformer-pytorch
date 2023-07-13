import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, xin):
        pass
    

class DotProductAttention(nn.Module):
    def __init__(self, dim_m=512, dim_q=512, dim_k=512, dim_v=512, masked=False) -> None:
        super().__init__()
        self.w_q = nn.Linear(in_features=dim_m, out_features=dim_q)
        self.w_k = nn.Linear(in_features=dim_m, out_features=dim_k)
        self.w_v = nn.Linear(in_features=dim_m, out_features=dim_v)
        self.softmax = nn.Softmax(dim=-1)
        self.masked = masked
    
    def forward(self, q_in, k_in, v_in):
        q, k, v = self.w_q(q_in), self.w_k(k_in), self.w_v(v_in)
        qk = q.bmm(k.transpose(1,2))
        dk_scale = math.sqrt(k.size(-1)) # k.size(-1) ** 0.5
        
        attn_aux = qk/dk_scale
        if self.masked:
            attn_aux = torch.tril(attn_aux, diagonal=0) #triangular inferior
            
        attn = self.softmax(attn_aux)
        return attn.bmm(v)  
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, dim_m=512, dim_k=512, dim_q=512, dim_v=512, masked=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList([DotProductAttention(dim_m=dim_m, dim_k=dim_k, dim_q=dim_q,
                                                        dim_v=dim_v, masked=masked) * n_heads])
        self.w_o = nn.Linear(in_features=n_heads*dim_v, out_features=dim_m)
    
    def forward(self, q_in, k_in, v_in):
        concat = torch.concat([head(q_in, k_in, v_in) for head in self.heads], dim=-1)
        return self.w_o(concat)
    

# ffn(x) = max(0, xW1 + b1)W2 + b2
class FeedForward(nn.Module):
    def __init__(self, dim_m=512, dim_ff=2048) -> None:
        super().__init__()
        self.w_1 = nn.Linear(in_features=dim_m, out_features=dim_ff)
        self.w_2 = nn.Linear(in_features=dim_ff, out_features=dim_m)
        self.relu = nn.ReLU()
    
    def forward(self, xin):
        return self.w_2(self.relu(self.w_1(xin)))
    
class ResidualConnection(nn.Module):
    def __init__(self, module, dropout=0.1) -> None:
        super().__init__()
        self.module = module
        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, xin):
        out = self.dropout(self.module(*xin))
        return self.layer_norm(xin[0] + out)
    
    
class EncoderLayer(nn.Module):
    def __init__(self, dim_m=512, n_heads=6, dim_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.attn = ResidualConnection(MultiHeadAttention(n_heads=n_heads), dropout=dropout)
        self.ff = ResidualConnection(FeedForward(dim_m=dim_m, dim_ff=dim_ff), dropout=dropout)
    
    def forward(self, xin):
        attn_out = self.attn(xin, xin, xin)
        ff_out = self.ff(attn_out)
        return ff_out

class EncoderModule(nn.Module):
    def __init__(self, n_encoders=6, dim_m=512, n_heads=6, dim_ff=2048, dropout=0.1) -> None:
        super().__init__()
        dim_k = dim_q = dim_m / n_heads 
        dim_v = dim_k
        self.encoders = nn.ModuleList([EncoderLayer(dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff, 
                                                    dim_q=dim_q, dim_k=dim_k, dim_v=dim_v, dropout=dropout) * n_encoders])
    
    def forward(self, xin):
        temp = xin
        for encoder in self.encoders:
            temp = encoder(temp)
        return temp
    
class DecoderLayer(nn.Module):
    def __init__(self, dim_m=512, n_heads=6, dim_ff=2048, dropout=0.1) -> None:
        self.masked_attn = ResidualConnection(MultiHeadAttention(n_heads=n_heads, masked=True), dropout=dropout)
        self.attn = ResidualConnection(MultiHeadAttention(n_heads=n_heads), dropout=dropout)
        self.ff = ResidualConnection(FeedForward(dim_m=dim_m, dim_ff=dim_ff), dropout=dropout)
        pass
    
    def forward(self, xin, memory):
        attn_masked_out = self.masked_attn(xin, xin, xin)
        attn_out = self.attn(memory, memory, attn_masked_out)
        return self.ff(attn_out)
                
class DecoderModule(nn.Module):
    def __init__(self, n_decoders=6, dim_m=512, n_heads=6, dim_ff=2048, dropout=0.1) -> None:
        super().__init__()
        dim_k = dim_q = dim_m / n_heads
        dim_v = dim_k
        self.decoders = nn.ModuleList([DecoderLayer(dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff, 
                                                    dim_q=dim_q, dim_k=dim_k, dim_v=dim_v, dropout=dropout) * n_decoders])
    
    def forward(self, xin, memory):
        temp = xin
        for decoder in self.decoders:
            temp = decoder(temp, memory)
        return temp
    
class Transformer(nn.Module):
    def __init__(self, n_encoders, n_decoders, n_heads, dim_m, dim_ff, dropout) -> None:
        super().__init__()
        self.encoders = EncoderModule(n_encoders=n_encoders, dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout)
        self.decoders = DecoderModule(n_decoders=n_decoders, dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout)
    
    def forward(self, xin):
        encoders_out = self.encoders(xin)
        decoders_out = self.decoders(xin, encoders_out)
        
        return decoders_out