import torch
import torch.nn as nn
from .commons import DecoderBlock, TransformerBlock, Decoder, SelfAttention, Encoder


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tag_vocab_size,
                 src_pad_idx,
                 tag_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device='cuda:0' if torch.cuda.is_available() else 'cpu',
                 max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size=src_vocab_size,
                               embed_size=embed_size,
                               number_of_layers=num_layers,
                               heads=heads,
                               device=device,
                               forward_expansion=forward_expansion,
                               dropout=dropout,
                               max_length=max_length)

        self.decoder = Decoder(
            tag_vocab_size=tag_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length
        )

        self.src_pad_idx = src_pad_idx
        self.tag_pad_idx = tag_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tag_mask(self, tag):
        n, tl = tag.shape
        tg_mask = torch.tril(torch.ones(tl, tl)).expand(
            n, 1, tl, tl
        )
        return tg_mask.to(self.device)

    def forward(self, src, tag):
        src_mask = self.make_src_mask(src=src)
        tag_mask = self.make_tag_mask(tag=tag)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tag, enc_src, src_mask, tag_mask)
        return out
