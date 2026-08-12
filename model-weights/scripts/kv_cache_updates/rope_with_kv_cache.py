from typing import Any
from statistics import median

import torch
import torch.nn as nn
from torch.nn import functional as F

import time

BLOCK_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 2500
EVAL_INTERVAL = 300
LEARNING_RATE = 3e-4
VOCAB_SIZE = 65
EVAL_ITERS = 200

N_EMBD = 384
N_HEADS = 6
HEAD_SIZE = N_EMBD // N_HEADS
N_LAYERS = 6

DROPOUT_RATE = 0.2

MAX_TOKENS_GENERATE = 500
BENCHMARK_TRIALS = 10

MAX_SEQ_LENGTH = BLOCK_SIZE + MAX_TOKENS_GENERATE

# ---------------

# the fact that we are using the triangular mask of the lower-triangular matrix is 
# the indication that we have a decoder-only network, no encoder. 

torch.manual_seed(1337)

with open("data/input.txt", "r") as file: 
    text = file.read()

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

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
    return x.to(DEVICE), y.to(DEVICE)

@torch.inference_mode()
def evaluate_loss(model): 
    #  over here, we collect a bunch of training averages.
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS).to(device=DEVICE)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss, _, _ = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class RoPEModule(nn.Module):
    def __init__(self, d_head, max_seq_length=MAX_TOKENS_GENERATE, base=10000):
        super().__init__()

        theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head)) # (d_head // 2,)

        seq_length = torch.arange(max_seq_length).float() # (max_seq_length,)
        theta_values = torch.outer(seq_length, theta) # (max_seq_length, d_head//2) -> gives you the rotation values  

        self.register_buffer("sin", torch.sin(theta_values))
        self.register_buffer("cos", torch.cos(theta_values))

    def forward(self, x, start_pos):
        T = x.shape[1]

        x_evens = x[..., 0::2]
        x_odds = x[..., 1::2]

        cos = self.cos[start_pos: start_pos + T, :].unsqueeze(0)
        sin = self.sin[start_pos: start_pos + T, :].unsqueeze(0)

        x_first = x_evens * cos - x_odds * sin
        x_second = x_evens * sin + x_odds * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_first
        out[..., 1::2] = x_second

        return out 

    
# create a `Head` module that performs one operation of self-attention
# NOTE: LM-Head is the head that turns embedding size vector into actual vocab size vector, a SA-head is the head the performs
# self-attention

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size

        # self.key = nn.Linear(N_EMBD, self.head_size, bias=False)
        # self.query = nn.Linear(N_EMBD, self.head_size, bias=False)
        # self.value = nn.Linear(N_EMBD, self.head_size)
        # vectorized projections for qkv
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.rope = RoPEModule(self.head_size, MAX_SEQ_LENGTH)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.register_buffer("tril", torch.tril(torch.ones(MAX_SEQ_LENGTH, MAX_SEQ_LENGTH, dtype=torch.int)).to(device=DEVICE))
    
    def forward(self, x, kv_cache, use_cache, cache_position=0):
        
        _,T,_ = x.shape
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = self.rope(k, cache_position)
        q = self.rope(q, cache_position)

        key_length = 0

        if use_cache: 
            kv_cache[0][:, cache_position: cache_position + T, :] = k
            kv_cache[1][:, cache_position: cache_position + T, :] = v

            key_length = cache_position + T
            k_attention = kv_cache[0][:, :key_length, :]
            v_attention = kv_cache[1][:, :key_length, :]
        else:
            key_length = T
            k_attention = k
            v_attention = v
        
        wei = q @ torch.transpose(k_attention, -2, -1) * self.head_size ** -0.5

        key_length = cache_position + T
        mask = self.tril[
            cache_position : cache_position + T,
            :key_length,
        ]
        wei = wei.masked_fill(mask==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v_attention
        
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
# essentially it 1s a module list and then takes two parameters: n_heads and n_embd. 
# and then it returns a concatenated version of all heads in the `forward` pass

class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size) for _ in range(n_heads)) # (B, T, 32)
        self.proj = nn.Linear(head_size * n_heads, N_EMBD) # (B, T, 32)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # a projection has NOTHING to do with residuals; all it does it provide some kind of linear transformation
        # for instance, if you want to downsample because you just expanded to large dimension, and then bringing it back down to lower dimension for processing purposes
        # or, in this case, using it to mix together embeddings with same-dimension mapping
    
    def forward(self, x, kv_cache, use_cache, cache_position):
        # currently kv_cache follows a shape of (B, N_HEAD, T, N_EMBD)
        # each head call returns an output of size (B, T, N_EMBD//N_HEAD) and a kv_cache of size (B, T, N_EMBD)
        # we need one big output which is the combination of all outputs from each head call, concatenated such that
        # resulting shape is (B, T, N_EMBD) and kv_cache[:, i, :, :] = result_kv_cache for each i in N_HEAD

        total_output = []

        for i, head in enumerate(self.heads):
            output = head(x, [kv_cache[0][:, i, :, :] if use_cache else None, kv_cache[1][:, i, :, :]] if use_cache else None, use_cache, cache_position)
            total_output.append(output)

        x = torch.cat(total_output, dim=-1)
        x = self.dropout(self.proj(x))

        return x # (B, N_HEAD, T, C)

# create a `Block` class that performs communication then computation (attention --> FFN)
class Block(nn.Module):
    def __init__(self, n_heads, head_size, embd_size):
        super().__init__()

        self.n_heads = n_heads
        self.head_size = head_size
        self.embd_size = embd_size

        self.attention = MultiHead(self.n_heads, self.head_size)
        self.ffn = FFN(self.embd_size) # (B, T, 32)
        self.ln1 = nn.LayerNorm(self.embd_size) # for self.attention
        self.ln2 = nn.LayerNorm(self.embd_size) # for ffn
    
    def forward(self, x, kv_cache, use_cache, cache_position):
        B,T,_ = x.shape
        out = self.attention(self.ln1(x), kv_cache, use_cache, cache_position)
        x = out + x # we add `x` back to mark a residual connection; essentially, we are mixing the old information with the new
        x = self.ffn(self.ln2(x)) + x # same here 
        return x # (B, N_HEAD, T, C)

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.table_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD) # we represent each token in each batch with just 32 values
        self.blocks = nn.ModuleList([Block(N_HEADS, HEAD_SIZE, N_EMBD) for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        
        self.tbt_total = 0.0
        self.ttft = 0.0
        self.tbt = 0.0
    
    def forward(self, index, targets=None, kv_cache=None, use_cache=False, cache_position=0):
        B, T = index.shape

        tok_emb = self.table_embedding(index) # (B, T, C), C = N_EMBD
        if use_cache and kv_cache is None:
            kv_cache = [None] * 2
            kv_cache[0] = torch.empty(size=(B, N_LAYERS, N_HEADS, MAX_SEQ_LENGTH, HEAD_SIZE), device=DEVICE)
            kv_cache[1] = torch.empty(size=(B, N_LAYERS, N_HEADS, MAX_SEQ_LENGTH, HEAD_SIZE), device=DEVICE)

        emb = tok_emb # (B, T, C)

        for i, block in enumerate(self.blocks):
            emb = block(emb, [kv_cache[0][:, i, :, :, :] if use_cache else None, kv_cache[1][:, i, :, :, :]] if use_cache else None, use_cache, cache_position)

        logits = self.lm_head(self.ln(emb)) # (B, T, VOCAB_SIZE) --> this is the new vector. `lm_head` is a layer that produces a 65-sized vector that shows
        # probabilities for what will be the next token. then, we just take the representation
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, kv_cache, cache_position + T


    @torch.inference_mode()
    def generate(self, index, max_tokens_generate, use_cache=False):
        self.tbt_total = 0.0
        self.ttft = 0.0
        
        times = []

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elif DEVICE == "mps":
            torch.mps.synchronize()


        start = time.perf_counter()
        B, T = index.shape
        was_training = self.training
        self.eval()

        kv_cache = None

        if use_cache:
            kv_cache = [None] * 2
            kv_cache[0] = torch.empty(size=(B, N_LAYERS, N_HEADS, MAX_SEQ_LENGTH, HEAD_SIZE), device=DEVICE)
            kv_cache[1] = torch.empty(size=(B, N_LAYERS, N_HEADS, MAX_SEQ_LENGTH, HEAD_SIZE), device=DEVICE)

        # prefill()
        logits, _, kv_cache, cache_position = self(index, kv_cache=kv_cache, use_cache=use_cache, cache_position=0)
        # first, get the actual raw values or `logits`
        for step in range(max_tokens_generate):
            next_token_logits = logits[:, -1, :] # this will take just the last timestep for all batches with all vocab size --> (B, C)
            # convert into probabilities from embeddings; we don't want raw logits, we want the converted probabilities or likelihood of what is the next character
            # probs = F.softmax(next_token_logits, dim=-1) # still (B, C)
            # next_idx = torch.multinomial(probs, 1) # then this becomes just (B, 1)
            next_idx = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            index = torch.cat((index, next_idx), dim=1)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elif DEVICE == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter())

            if step + 1 < max_tokens_generate:
                if use_cache:
                    logits, _, kv_cache, cache_position = self(
                        next_idx,
                        kv_cache=kv_cache,
                        use_cache=True,
                        cache_position=cache_position
                    )
                else:
                    logits, _, _, _ = self(
                        index,
                        use_cache=False,
                    )

        if times:
            self.ttft = times[0] - start

        if len(times) > 1:
            self.tbt = sum(
                t2 - t1 for t1, t2 in zip(times, times[1:])
            ) / (len(times) - 1)
        if was_training:
            self.train()
        return index
    
    def run_train(self, optimizer): # we want to optimize this training loop such that we are averaging out losses over multiple eval cycles. for instance, we can iterate for both training and val, and then get the thing
        for i in range(EPOCHS):
            if i % EVAL_INTERVAL == 0:
                out = evaluate_loss(self)
                print(f'Epoch {i}: train loss (%.4f) {out["train"]:.4f}, val loss (%.4f) {out["val"]:.4f}')
            xb, yb = get_batch("train")

            _, loss, _, _ = self(xb, yb)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

m = TransformerLanguageModel().to(device=DEVICE)
optimizer = torch.optim.AdamW(params=m.parameters(), lr=LEARNING_RATE)
m.run_train(optimizer)

prompt = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=DEVICE,
)

m.generate(prompt.clone(), MAX_TOKENS_GENERATE, use_cache=True)
m.generate(prompt.clone(), MAX_TOKENS_GENERATE, use_cache=False)

benchmark_results = {
    "cached": {"tbt": [], "ttft": []},
    "uncached": {"tbt": [], "ttft": []},
}

generated_outputs = {}

for trial in range(BENCHMARK_TRIALS):
    modes = [True, False] if trial % 2 == 0 else [False, True]

    for use_cache in modes:
        mode = "cached" if use_cache else "uncached"
        generated_outputs[mode] = m.generate(
            prompt.clone(),
            MAX_TOKENS_GENERATE,
            use_cache=use_cache,
        )
        benchmark_results[mode]["tbt"].append(m.tbt)
        benchmark_results[mode]["ttft"].append(m.ttft)

with open("/outputs/rope_cache_comparison.txt", "w") as f:
    outputs_equal = torch.equal(
        generated_outputs["cached"],
        generated_outputs["uncached"],
    )
    f.write(f"outputs_equal={outputs_equal}\n")
    f.write(f"benchmark_trials={BENCHMARK_TRIALS}\n")
    f.write(f"max_tokens_generated={MAX_TOKENS_GENERATE}\n")
    f.write(
        f"cached_tbt_median={median(benchmark_results['cached']['tbt'])}\n"
    )
    f.write(
        f"cached_ttft_median={median(benchmark_results['cached']['ttft'])}\n"
    )
    f.write(f"cached_tbt_trials={benchmark_results['cached']['tbt']}\n")
    f.write(f"cached_ttft_trials={benchmark_results['cached']['ttft']}\n")
    f.write("-----------------------\n")
    f.write(
        f"uncached_tbt_median={median(benchmark_results['uncached']['tbt'])}\n"
    )
    f.write(
        f"uncached_ttft_median={median(benchmark_results['uncached']['ttft'])}\n"
    )
    f.write(f"uncached_tbt_trials={benchmark_results['uncached']['tbt']}\n")
    f.write(f"uncached_ttft_trials={benchmark_results['uncached']['ttft']}\n")
    f.write("-----------------------\n")
