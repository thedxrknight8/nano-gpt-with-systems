import torch 
import torch.nn as nn
from torch.nn import functional as F

from config.config import A100ConfigInference, A100ConfigTrain

config = A100ConfigInference()

class RoPEModule(nn.Module):
    def __init__(self, d_head, max_seq_length, base=10000):
        super().__init__()

        theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head)) # (d_head // 2,)

        seq_length = torch.arange(max_seq_length).float() # (max_seq_length,)
        theta_values = torch.outer(seq_length, theta) # (max_seq_length,d_head//2) -> gives you the rotation values  

        self.register_buffer("sin", torch.sin(theta_values))
        self.register_buffer("cos", torch.cos(theta_values))

    def forward(self, x):
        T = x.shape[1]

        x_evens = x[..., 0::2]
        x_odds = x[..., 1::2]

        cos = self.cos[:T, :].unsqueeze(0)
        sin = self.sin[:T, :].unsqueeze(0)

        x_first = x_evens * cos - x_odds * sin
        x_second = x_evens * sin + x_odds * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_first
        out[..., 1::2] = x_second

        return out 
        

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size)
        self.rope = RoPEModule(self.head_size, config.block_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.int)).to(
                device=config.device
            ),
        )

    def forward(self, x):

        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        q = self.rope(q)
        k = self.rope(k)

        wei = q @ torch.transpose(k, -2, -1) * self.head_size**-0.5
        wei = torch.masked_fill(wei, (self.tril[:T, :T] == 0), float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v


class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(n_heads))
        self.proj = nn.Linear(head_size * n_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class Block(nn.Module):
    def __init__(self, n_heads, head_size, embd_size):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.embd_size = embd_size

        self.attention = MultiHead(self.n_heads, self.head_size)
        self.ffn = FFN(self.embd_size)
        self.ln1 = nn.LayerNorm(self.embd_size)
        self.ln2 = nn.LayerNorm(self.embd_size)

    def forward(self, x):
        out = self.attention(self.ln1(x))
        x = out + x
        x = self.ffn(self.ln2(x)) + x
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.table_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config.n_heads, config.head_size, config.n_embd) for _ in range(config.n_layers)]
        )
        self.ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.table_embedding(index)
        emb = tok_emb

        for block in self.blocks:
            emb = block(emb)
        logits = self.lm_head(self.ln(emb))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    