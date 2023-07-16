from transformer import Transformer, PositionalEncoding
import torch
def main():
    transformer = Transformer(n_encoders=1, n_decoders=1, in_len=32, feat_in_enc=12, feat_in_dec=1, n_heads=8, dim_m=512, out_len=10)
    src = torch.rand(64, 32, 12)
    tgt = torch.rand(64, 1, 1)
    out = transformer(src, tgt)
    print(out.size())
    
if __name__ == "__main__":
    main()