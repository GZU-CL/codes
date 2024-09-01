### python:3.9
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length, device):
        super(PositionalEncoding, self).__init__()
        self.device = device

        # 创建一个max_lengthxembed_size的矩阵
        position_encoding = torch.zeros(max_length, embed_size)

        for pos in range(max_length):
            for i in range(0, embed_size, 2):
                position_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                if i + 1 < embed_size:
                    position_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        # 为位置编码增加一个批次维度，并冻结参数
        self.position_encoding = position_encoding.unsqueeze(0).to(self.device)

    def forward(self, x):
        # 将位置编码添加到词嵌入上
        return x + self.position_encoding[:, :x.size(1), :]

# 多头自注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert (self.heads_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.heads_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.heads_dim)
        out = self.fc_out(out)

        return out

# 编码器层
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward_expansion * embed_size, embed_size))
        self.dropout = dropout

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 编码器
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [EncoderBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion) for _ in
             range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        out = self.dropout(self.position_encoding(self.word_embedding))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# 解码器层
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = dropout

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.encoder_block(value, key, query, src_mask)
        return out

# 解码器
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(position))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)

        return out

# 创建编码器-解码器类
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size=256, num_layers=6, forward_expansion=4, heads=8,
                 dropout=0.1, device='cuda', max_length=100):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout,
                               max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device,
                               max_length)
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
