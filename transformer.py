import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim_m) -> None:
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_m, 2) * (-math.log(10000.0) / dim_m))
        pe = torch.zeros(seq_len, dim_m)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, xin):
        return xin + self.pe[:xin.size(1)]
    

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
        dk_scale = k.size(-1) ** 0.5

        attn_aux = qk/dk_scale
        if self.masked:
            mask = torch.tril(torch.ones(attn_aux.size()), diagonal=0)
            mask_inf = (mask*torch.triu(torch.ones(attn_aux.size()) * -1.0, diagonal=1))
            attn_aux = attn_aux*mask + mask_inf

        attn = self.softmax(attn_aux)
        return attn.bmm(v)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, dim_m=512, dim_k=512, dim_q=512, dim_v=512, masked=False, *args, **kwargs) -> None:
        super().__init__()
        self.heads = nn.ModuleList([DotProductAttention(dim_m=dim_m, dim_k=dim_k, dim_q=dim_q,
                                                        dim_v=dim_v, masked=masked) for _ in range(n_heads)])
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
    def __init__(self, module, dim, dropout=0.1) -> None:
        super().__init__()
        self.module = module
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xin, residx=0):
        out = self.dropout(self.module(*xin))
        return self.layer_norm(xin[residx] + out)
    
    
class EncoderLayer(nn.Module):
    def __init__(self, dim_m=512, n_heads=6, dim_ff=2048, dim_q=512, dim_k=512, dim_v=512, dropout=0.1) -> None:
        super().__init__()
        self.attn = ResidualConnection(MultiHeadAttention(n_heads=n_heads, dim_m=dim_m, dim_k=dim_k, dim_q=dim_q, dim_v=dim_v, masked=False), dim=dim_m, dropout=dropout)
        self.ff = ResidualConnection(FeedForward(dim_m=dim_m, dim_ff=dim_ff), dim=dim_m, dropout=dropout)

    def forward(self, xin):
        attn_out = self.attn([xin, xin, xin])
        ff_out = self.ff([attn_out])
        return ff_out

class EncoderModule(nn.Module):
    def __init__(self, n_encoders=6, dim_m=512, n_heads=6, dim_ff=2048, dropout=0.1) -> None:
        super().__init__()
        dim_k = dim_q = int(dim_m / n_heads)
        dim_v = dim_k
        self.encoders = nn.ModuleList([EncoderLayer(dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff,
                                                    dim_q=dim_q, dim_k=dim_k, dim_v=dim_v, dropout=dropout) for _ in range(n_encoders)])

    def forward(self, xin):
        temp = xin
        for encoder in self.encoders:
            temp = encoder(temp)
        return temp
    
class DecoderLayer(nn.Module):
    def __init__(self, dim_m=512, n_heads=6, dim_ff=2048, dim_q=512, dim_k=512, dim_v=512, dropout=0.1) -> None:
        super().__init__()
        self.masked_attn = ResidualConnection(MultiHeadAttention(n_heads=n_heads, dim_m=dim_m, dim_k=dim_k, dim_q=dim_q, dim_v=dim_v, masked=True), dim=dim_m, dropout=dropout)
        self.attn = ResidualConnection(MultiHeadAttention(n_heads=n_heads, dim_m=dim_m, dim_k=dim_k, dim_q=dim_q, dim_v=dim_v, masked=False), dim=dim_m, dropout=dropout)
        self.ff = ResidualConnection(FeedForward(dim_m=dim_m, dim_ff=dim_ff), dim=dim_m, dropout=dropout)
        pass

    def forward(self, xin, memory):
        attn_masked_out = self.masked_attn([xin, xin, xin])
        attn_out = self.attn([attn_masked_out, memory, memory])
        return self.ff([attn_out])
                
class DecoderModule(nn.Module):
    def __init__(self, n_decoders=6, dim_m=512, n_heads=6, dim_ff=2048, dropout=0.1) -> None:
        super().__init__()
        dim_k = dim_q = int(dim_m / n_heads)
        dim_v = dim_k
        self.decoders = nn.ModuleList([DecoderLayer(dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff,
                                                    dim_q=dim_q, dim_k=dim_k, dim_v=dim_v, dropout=dropout) for _ in range(n_decoders)])

    def forward(self, xin, memory):
        temp = xin
        for decoder in self.decoders:
            temp = decoder(temp, memory)
        return temp
    
class Transformer(nn.Module):
    def __init__(self, n_encoders, n_decoders, n_heads, dim_m=512, dim_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.encoders = EncoderModule(n_encoders=n_encoders, dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout)
        self.decoders = DecoderModule(n_decoders=n_decoders, dim_m=dim_m, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout)

    def forward(self, src, target):
        encoders_out = self.encoders(src)
        decoders_out = self.decoders(target, encoders_out)
        return decoders_out

class TransformerModel(nn.Module):
  def __init__(self, n_encoders, n_decoders, n_heads, in_len, out_len, feat_in, dim_m=512, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.pe = PositionalEncoding(seq_len=max(out_len+1, in_len), dim_m=dim_m)
        self.enc_emb = nn.Linear(in_features=feat_in, out_features=dim_m)
        self.dec_emb = nn.Linear(in_features=feat_in, out_features=dim_m)
        self.transformer = Transformer(n_encoders=n_encoders, n_decoders=n_decoders, n_heads=n_heads, dim_m=dim_m, dim_ff=dim_ff, dropout=dropout)
        self.lineal = nn.Linear(in_features=dim_m, out_features=1)

  def forward(self, src, target):
        # todo: autorregresive prediction
        src_emb = self.pe(self.enc_emb(src))
        tgt_emb = self.pe(self.dec_emb(target))
        out = self.transformer(src_emb, tgt_emb)
        out = self.lineal(out)
        return out