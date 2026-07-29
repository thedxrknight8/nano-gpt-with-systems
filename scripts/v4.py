import torch
import torch.nn as nn
from torch.nn import functional as F

import time
import os

# hyperparameters
BLOCK_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 2500
EVAL_INTERVAL = 300
LEARNING_RATE = 3e-4
VOCAB_SIZE = 65
EVAL_ITERS = 200

N_EMBD = 384
N_HEADS = 6
# ATTENTION_SIZE = 16
HEAD_SIZE = N_EMBD // N_HEADS
N_LAYERS = 6

DROPOUT_RATE = 0.2

MAX_TOKENS_GENERATE = 250

# ---------------

# the fact that we are using the triangular mask of the lower-triangular matrix is 
# the indication that we have a decoder-only network, no encoder. 

torch.manual_seed(1337)

with open("data/input.txt", "r") as file: 
    text = file.read()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
# conversion sets for characters from token indexes
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# splitting the data in `train` and `test`
data = torch.tensor(encode(text), dtype=torch.long)
train = data[:int(0.9 * len(data))]
test = data[int(0.9 * len(data)):]

# create batches for specific splits
def get_batch(split, batch_size=BATCH_SIZE): 
    data = train if split == "train" else test
    ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate_loss(model): 
    #  over here, we collect a bunch of training averages.
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS).to(device=device)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a `Head` module that performs one operation of self-attention
# NOTE: LM-Head is the head that turns embedding size vector into actual vocab size vector, a SA-head is the head the performs
# self-attention

class AttentionHead(nn.Module):
    def __init__(self, head_size, num_head):
        super().__init__()

        self.head_size = head_size
        self.num_head = num_head
            
        self.key = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.query = nn.Linear(N_EMBD, self.head_size, bias=False)
        self.value = nn.Linear(N_EMBD, self.head_size)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE,dtype=torch.int)).to(device=device))
    
    def forward(self, x, kv_cache=torch.Tensor | None, use_cache=False):
        B,T,_ = x.shape
        q = self.query(x)
        
        wei = q @ torch.transpose(k, -2, -1) * self.head_size ** -0.5
        
        wei = torch.masked_fill(wei, (self.tril[:T, :T] == 0), float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        if use_cache:
            kv_cache[]
        
        return out # (B, T, 4)
        
class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # (B, T, 32 * 4)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # this is another projection; this time for the FFN, but for specifically upsampling to get a more rich representation
            nn.Dropout(DROPOUT_RATE)
        )
    
    def forward(self, x): 
        return self.net(x)

# create a Multi-Head Attention module
# essentially it uses a module list and then takes two parameters: n_heads and n_embd. 
# and then it returns a concatenated version of all heads in the `forward` pass

class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size, kv_cache, use_cache):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size, idx, kv_cache, use_cache) for idx in range(n_heads)) # (B, T, 32)
        self.proj = nn.Linear(head_size * n_heads, N_EMBD) # (B, T, 32)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # a projection has NOTHING to do with residuals; all it does it provide some kind of linear transformation
        # for instance, if you want to downsample because you just expanded to large dimension, and then bringing it back down to lower dimension for processing purposes
        # or, in this case, using it to mix together embeddings with same-dimension mapping
    
    def forward(self, x, kv_cache=None, use_cache=False):
        if use_cache:
            x = torch.cat([head(x, kv_cache, use_cache) for head in self.heads], dim=-1)
        else:
            x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))

# create a `Block` class that performs communication then computation (attention --> FFN)
class Block(nn.Module):
    def __init__(self, n_heads, head_size, embd_size):
        super().__init__()
        self.attention = MultiHead(n_heads, head_size)
        self.ffn = FFN(embd_size) # (B, T, 32)
        self.ln1 = nn.LayerNorm(embd_size) # for self.attention
        self.ln2 = nn.LayerNorm(embd_size) # for ffn
    
    def forward(self, x: torch.Tensor, use_cache=False):
        kv_cache = None
        if use_cache:
            B,T,_= x.shape
            kv_cache = torch.zeros(B, N_HEADS, T, HEAD_SIZE)
        x = self.attention(self.ln1(x), kv_cache, use_cache) + x # we add `x` back to mark a residual connection; essentially, we are mixing the old information with the new  
        x = self.ffn(self.ln2(x)) + x # same here 
        return x # (B, T, 32)

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.table_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD) # we represent each token in each batch with just 32 values
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[Block(n_heads=N_HEADS, head_size=HEAD_SIZE, embd_size=N_EMBD) for _ in range(N_LAYERS)]
        )
        self.ln = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        
        
        self.tbt_total = 0.0
        self.ttft = 0.0
        self.tbt = 0.0
    
    def forward(self, index, targets=None, use_cache=False) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = index.shape
        tok_emb = self.table_embedding(index) # (B, T, C), C = N_EMBD
        pos_emb = self.positional_embedding(torch.arange(end=T, device=index.device)) # (T, C), C = N_EMBD -> Broadcast operation applied!
        
        emb = tok_emb + pos_emb # (B, T, C)
        emb = self.blocks(emb, use_cache)
        logits = self.lm_head(self.ln(emb)) # (B, T, VOCAB_SIZE) --> this is the new vector. `lm_head` is a layer that produces a 65-sized vector that shows 
        # probabilities for what will be the next token. then, we just take the representation
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.inference_mode()
    def generate(self, index, max_tokens_generate):
        self.tbt_total = 0.0
        self.ttft = 0.0
        
        times = []
        start = time.perf_counter()
        self.eval()
        # first, get the actual raw values or `logits`
        for _ in range(max_tokens_generate):
            idx_cond = index[:, -BLOCK_SIZE:] # crop to last BLOCK_SIZE tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # this will take just the last timestep for all batches with all vocab size --> (B, C)
            # convert into probabilities from embeddings; we don't want raw logits, we want the converted probabilities or likelihood of what is the next character
            probs = F.softmax(logits, dim=-1) # still (B, C)
            next_idx = torch.multinomial(probs, 1) # then this becomes just (B, 1)
            
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
    
    def run_train(self, optimizer): # we want to optimize this training loop such that we are averaging out losses over multiple eval cycles. for instance, we can iterate for both training and val, and then get the thing
        for i in range(EPOCHS):
            if i % EVAL_INTERVAL == 0:
                out = evaluate_loss(self)
                print(f'Epoch {i}: train loss (%.4f) {out["train"]:.4f}, val loss (%.4f) {out["val"]:.4f}')
            xb, yb = get_batch("train")

            _, loss = self(xb, yb)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

m = TransformerLanguageModel().to(device=device)
optimizer = torch.optim.AdamW(params=m.parameters(), lr=LEARNING_RATE)
m.run_train(optimizer)

idxs = m.generate(get_batch("test", 1)[0], MAX_TOKENS_GENERATE)

for idx in idxs:
    print(decode(idx.tolist()))

with open("/outputs/metrics.txt", "w") as f:
    f.write(f"params={sum(p.numel() for p in m.parameters())}")
    f.write(f"average_tbt={m.tbt} max_tokens_generated={MAX_TOKENS_GENERATE}\n")
    f.write(f"ttft={m.ttft} max_tokens_generated={MAX_TOKENS_GENERATE}")
    f.write("-----------------------")

