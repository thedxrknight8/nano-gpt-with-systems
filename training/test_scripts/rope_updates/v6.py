import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class RoPEModule(nn.Module):
    def __init__(self, d_head, max_seq_length=MAX_TOKENS_GENERATE, base=10000):
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
        self.key = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.query = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.value = nn.Linear(N_EMBD, self.head_size)
        self.rope = RoPEModule(self.head_size, MAX_TOKENS_GENERATE)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.int)).to(
                device=DEVICE
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
            nn.Dropout(DROPOUT_RATE),
        )

    def forward(self, x):
        return self.net(x)


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(n_heads))
        self.proj = nn.Linear(head_size * n_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT_RATE)

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

        self.table_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.blocks = nn.ModuleList(
            [Block(N_HEADS, HEAD_SIZE, N_EMBD) for _ in range(N_LAYERS)]
        )
        self.ln = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

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

    @torch.inference_mode()
    def generate(self, index, max_tokens_generate):
        was_training = self.training
        self.eval()

        prompt = index[:, -BLOCK_SIZE:]
        logits, _ = self(prompt)

        for step in range(max_tokens_generate):
            next_token_logits = logits[:, -1, :]
            next_token_logits = F.softmax(next_token_logits, dim=-1)
            next_idx = torch.multinomial(
                next_token_logits,
                num_samples=1,
            )

            index = torch.cat((index, next_idx), dim=1)

            if step + 1 < max_tokens_generate:
                idx_cond = index[:, -BLOCK_SIZE:]
                logits, _ = self(idx_cond)

        if was_training:
            self.train()
        return index

    def run_train(self, optimizer):
        for i in range(EPOCHS):
            if i % EVAL_INTERVAL == 0:
                out = evaluate_loss(self)
                print(
                    f'Epoch {i}: train loss (%.4f) {out["train"]:.4f}, val loss (%.4f) {out["val"]:.4f}'
                )
            xb, yb = get_batch("train")

            _, loss = self(xb, yb)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    m = TransformerLanguageModel().to(device=DEVICE)
    optimizer = torch.optim.AdamW(params=m.parameters(), lr=LEARNING_RATE)
    m.run_train(optimizer)

    prompt = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=DEVICE,
    )

    torch.save({
        "model_save_dict": m.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "BLOCK_SIZE": BLOCK_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "N_EMBD": N_EMBD,
            "N_HEADS": N_HEADS,
            "N_LAYERS": N_LAYERS
        }
    }, "/outputs/nanogpt.pt")

    generated_output = m.generate(prompt, MAX_TOKENS_GENERATE)
    print(decode(generated_output[0].tolist()))
