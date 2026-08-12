import torch
import torch.nn as nn
from torch.nn import functional as F

import time
import os

BLOCK_SIZE = 8
BATCH_SIZE = 3
EPOCHS = 2500
EVAL_INTERVAL = 300
LEARNING_RATE = 3e-4
VOCAB_SIZE = 65
EVAL_ITERS = 200

N_EMBD = 32
N_HEADS = 1
HEAD_SIZE = N_EMBD // N_HEADS
N_LAYERS = 1

DROPOUT_RATE = 0.2

MAX_TOKENS_GENERATE = 250

torch.manual_seed(1337)

with open("data/input.txt", "r") as file: 
    text = file.read()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
train = data[:int(0.9 * len(data))]
test = data[int(0.9 * len(data)):]

def get_batch(split, batch_size=BATCH_SIZE): 
    data = train if split == "train" else test
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate_loss(model): 
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS).to(device=device)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss, _, _ = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class AttentionHead(nn.Module):
    def __init__(self, head_size, head_idx):
        super().__init__()

        self.head_size = head_size
        self.head_idx = head_idx
            
        self.key = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.query = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.value = nn.Linear(N_EMBD, self.head_size)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE,dtype=torch.int)).to(device=device))
    
    def forward(self, x, k_cache=None, v_cache=None, use_cache=False, mask_use=True):
        _,T,_ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        k_attention = k
        v_attention = v
        
        if use_cache:
            if T - BLOCK_SIZE >= 0:
                k_cache = k[:, (T-BLOCK_SIZE):, :]
                v_cache = v[:, (T-BLOCK_SIZE):, :]
            elif k_cache is None or v_cache is None or k_cache.shape[-2] == v_cache.shape[-2] == 0:
                k_cache = k
                v_cache = v
            elif k_cache.shape[1] + T > BLOCK_SIZE:
                k_cache = k_cache[:, (T-BLOCK_SIZE):, :]
                k_cache = torch.cat([k_cache, k], dim=-2)
                
                v_cache = v_cache[:, (T-BLOCK_SIZE):, :]
                v_cache = torch.cat([v_cache, v], dim=-2)
            else:
                k_cache = torch.cat([k_cache, k[:, -BLOCK_SIZE:, :]], dim=-2)
                v_cache = torch.cat([v_cache, v[:, -BLOCK_SIZE:, :]], dim=-2)
            
            k_attention = k_cache
            v_attention = v_cache
        
        wei = q @ torch.transpose(k_attention, -2, -1) * self.head_size ** -0.5
        if mask_use:
            wei = torch.masked_fill(wei, (self.tril[:T, :T] == 0), float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v_attention
        
        return out, k_cache, v_cache
        
class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT_RATE)
        )
    
    def forward(self, x): 
        return self.net(x)

class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size, idx) for idx in range(n_heads))
        self.proj = nn.Linear(head_size * n_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, x, k_cache=None, v_cache=None, use_cache=False, mask_use=True):
        x = torch.cat([head(x, k_cache=k_cache, v_cache=v_cache, use_cache=use_cache, mask_use=mask_use) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))

class Block(nn.Module):
    def __init__(self, n_heads, head_size, embd_size, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = MultiHead(n_heads, head_size)
        self.ffn = FFN(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)
    
    def forward(self, x: torch.Tensor, use_cache=False, k_cache=None, v_cache=None, mask_use=True, ):
        layer_k_cache = k_cache[:, self.layer_idx, :, :, :]
        layer_v_cache = v_cache[:, self.layer_idx, :, :, :]
        out, updated_layer_k, updated_layer_v = self.attention(self.ln1(x), layer_k_cache, layer_v_cache, use_cache, mask_use)
        x = x + out
        x = self.ffn(self.ln2(x)) + x
        return x, updated_layer_k, updated_layer_v

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.table_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.block = nn.Sequential(*[Block(N_HEADS, HEAD_SIZE, N_EMBD, layer_idx=i) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        
        self.tbt_total = 0.0
        self.ttft = 0.0
        self.tbt = 0.0
    
    def forward(self, index, targets=None, use_cache=False, mask_use=True) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = index.shape
        tok_emb = self.table_embedding(index)
        past_length = None
        k_cache = v_cache = None
        if use_cache:
            past_length = 0
            if k_cache is not None:
                past_length = k_cache.shape[-2]
            else:
                k_cache = torch.empty(size=(B, N_LAYERS, N_HEADS, T, HEAD_SIZE), device=index.device)
                v_cache = torch.empty(size=(B, N_LAYERS, N_HEADS, T, HEAD_SIZE), device=index.device)

            positions = torch.arange(past_length, past_length + T, device=index.device)
            pos_emb = self.positional_embedding(positions)
            emb = tok_emb + pos_emb
            emb, k_cache, v_cache = self.block(emb, use_cache, k_cache, v_cache, mask_use=mask_use)
        else:
            pos_emb = self.positional_embedding(torch.arange(end=T, device=index.device))
            emb = tok_emb + pos_emb
            emb, _, _ = self.block(emb, use_cache, mask_use=mask_use)
        logits = self.lm_head(self.ln(emb))
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, k_cache, v_cache

    @torch.inference_mode()
    def generate(self, index, max_tokens_generate):
        assert max_tokens_generate > 0
            
        self.tbt_total = 0.0
        self.ttft = 0.0
        
        B,_ = index.shape
        
        times = []
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        self.eval()
        use_cache = True
        
        if use_cache:
            logits, _, k_cache, v_cache = self(index, targets=None, use_cache=True) 
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)

            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            
            now = time.perf_counter()
            times.append(now)
            
            index = torch.cat((index, next_idx), dim=1)

            for _ in range(max_tokens_generate - 1):
                logits, _, k_cache, v_cache = self(next_idx, use_cache=True, mask_use=True)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()
                
                now = time.perf_counter()
                times.append(now)
                
                index = torch.cat((index, next_idx), dim=1)
                
                if k_cache.shape[-2] >= BLOCK_SIZE:
                    break
        else:
            for _ in range(max_tokens_generate):
                idx_cond = index[:, -BLOCK_SIZE:] 
                logits, _, _, _ = self(idx_cond)
                logits = logits[:, -1, :] 

                probs = F.softmax(logits, dim=-1) 
                next_idx = torch.multinomial(probs, 1)

                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()

                now = time.perf_counter()
                times.append(now)

                index = torch.cat((index, next_idx), dim=1)
                
        self.ttft = times[0] - start
        self.tbt = sum(t2 - t1 for t1, t2 in zip(times, times[1:])) / max(1, len(times) - 1)
        return index
    
    def run_train(self, optimizer):
        for i in range(EPOCHS):
            if i % EVAL_INTERVAL == 0:
                out = evaluate_loss(self)
                print(f'Epoch {i}: train loss (%.4f) {out["train"]:.4f}, val loss (%.4f) {out["val"]:.4f}')
            xb, yb = get_batch("train")

            _, loss, _, _ = self(xb, yb)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

m = TransformerLanguageModel().to(device=device)
optimizer = torch.optim.AdamW(params=m.parameters(), lr=LEARNING_RATE)
m.run_train(optimizer)

idxs = m.generate(get_batch("test", BATCH_SIZE)[0][:, :3], MAX_TOKENS_GENERATE)

# for idx in idxs:
#     print(decode(idx.tolist()))

print(m.tbt)
