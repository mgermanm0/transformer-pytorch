from transformer import Transformer, TransformerModel, PositionalEncoding
import torch
def main():
    transformer = TransformerModel(n_encoders=1, n_decoders=1, in_len=32, feat_in=512, n_heads=8, dim_m=512, out_len=10)
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 10, 512)
    out = transformer(src, tgt)
    print(out.size())
    
    pec = PositionalEncoding(32, 512)
    print(src)
    outpe = pec(src)
    print(outpe)
    
if __name__ == "__main__":
    main()