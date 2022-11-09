import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self,
                 embed_size,
                 heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dims = embed_size // heads

        assert (self.head_dims * self.heads == self.embed_size), "Embed Size Need to be div by heads"
        self.values = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=False)
        self.keys = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=False)
        self.queries = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=False)
        self.fc_out = nn.Linear(in_features=self.head_dims * heads, out_features=self.embed_size, bias=True)

    def forward(self,
                value,
                key,
                query,
                mask):
        n = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        value = value.reshape(n, value_len, self.heads, self.head_dims)
        query = value.reshape(n, query_len, self.heads, self.head_dims)
        key = value.reshape(n, key_len, self.heads, self.head_dims)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = nn.Softmax(energy / (self.embed_size ** (1, 2)), dim=3)
        out = torch.einsum('nhql,nlhq->nqhd', [attention, value]).reshape(
            n, query_len, self.heads * self.head_dims
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size,
                 heads,
                 dropout,
                 forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.heads = heads
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 number_of_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size,
                                 dropout=dropout,
                                 forward_expansion=forward_expansion,
                                 heads=heads)
                for _ in range(number_of_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(n, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size,
                 heads,
                 forward_expansion,
                 dropout,
                 device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tag_mask):
        attention = self.attention(x, x, x, tag_mask)
        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 tag_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tag_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size=embed_size, heads=heads, forward_expansion=forward_expansion, dropout=dropout,
                          device=device) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, tag_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tag_mask):
        n, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(n, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tag_mask)

        out = self.fc_out(x)
        return out
