import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.lib import make_pad_mask, make_subsequent_mask


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # начальная матрица
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe) # надо для GPU и сохранения модели

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(self.d_k * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k = None, v = None, mask = None):
        # q, k, v: (batch, seq, d_model)
        if k is None:
            k = q
        if v is None:
            v = q

        batch = q.size(0)
        # view для разбиения на головы, transpose для удобного формата
        q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_q, d_k)
        k = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_k, d_k)
        v = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_k, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, seq_q, seq_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1) # dim=-1 - последняя размерность
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (batch, heads, seq_q, d_k)
        # contiguous для работы view(и для копирования)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.d_model)  # (batch, seq_q, d_model)
        out = self.W_o(context)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x2 = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + x2)

        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        t2 = self.masked_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + t2)

        t2 = self.cross_attn(tgt, memory, memory, mask=src_mask)
        tgt = self.norm2(tgt + t2)

        t2 = self.ff(tgt)
        tgt = self.norm3(tgt + t2)
        return tgt

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_encoder_layers,
        n_decoder_layers,
        max_len,
        pad_idx,
        dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoder(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)])

        self.output_linear = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        # src: (batch, src_len)
        src_mask = make_pad_mask(src, self.pad_idx)  # (batch, 1, 1, src_len)
        x = self.tok_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        return x, src_mask

    def decode(self, tgt, memory, src_mask, tgt_pad_mask=None):
        # tgt: (batch, tgt_len)
        batch, tgt_len = tgt.shape
        tgt_mask_sub = make_subsequent_mask(tgt_len).to(tgt.device)  # (1, 1, tgt_len, tgt_len)
        if tgt_pad_mask is None:
            tgt_pad_mask = make_pad_mask(tgt, self.pad_idx)  # (batch, 1, 1, tgt_len)
        # to (batch, 1, tgt_len, tgt_len)
        tgt_mask = (tgt_pad_mask & tgt_mask_sub.to(tgt_pad_mask.dtype)).to(tgt.device)

        y = self.tok_embedding(tgt) * math.sqrt(self.d_model)
        y = self.pos_encoder(y)
        for layer in self.decoder_layers:
            y = layer(y, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return y

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        out = self.decode(tgt, memory, src_mask)
        logits = self.output_linear(out)
        return logits
