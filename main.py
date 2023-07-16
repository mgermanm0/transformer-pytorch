from transformer import Transformer, TransformerModel
import torch
def main():
    transformer = TransformerModel(n_encoders=6, n_decoders=6, n_heads=8, dim_m=512, out_len=1)
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 1, 512)
    out = transformer(src, tgt)
    print(out.size())

if __name__ == "__main__":
    main()