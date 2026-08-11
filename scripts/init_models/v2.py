import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
BLOCK_SIZE = 8
BATCH_SIZE = 4
EPOCHS = 20000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3
VOCAB_SIZE = 65
EVAL_ITERS = 200
N_EMBD = 32
HEAD = 16
# ---------------

torch.manual_seed(1337)

with open("data/input.txt", "r") as file: 
    text = file.read()

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
def get_batch(split): 
    data = train if split == "train" else test
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    return x, y

@torch.no_grad()
def evaluate_loss(model): 
    #  over here, we collect a bunch of training averages.
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

# create a `Head` module that performs one operation of self-attention
# NOTE: LM-Head is the head that turns embedding size vector into actual vocab size vector, a SA-head is the head the performs
# self-attention

class Head(nn.Module):
    def __init__(self, h_embd=HEAD):
        super().__init__()

        self.head_size = h_embd
        self.key = nn.Linear(N_EMBD, h_embd, bias=False)
        self.query = nn.Linear(N_EMBD, h_embd, bias=False)
        self.value = nn.Linear(N_EMBD, h_embd)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE,BLOCK_SIZE,dtype=torch.int)))
    
    def forward(self, x):
        _,T,_ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ torch.transpose(k, -2, -1) * self.head_size ** -0.5
        
        wei = torch.masked_fill(wei, (self.tril[:T, :T] == 0), float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        
        return out
    
# create a Multi-Head Attention module
# essentially it uses a module list and then takes two parameters: n_heads and h_embd. 
# and then it returns a concatenated version of all heads in the `forward` pass

class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHead(nn.Module):
    def __init__(self, n_heads, h_embd):
        super().__init__()
        self.heads = nn.ModuleList(Head(h_embd) for _ in range(n_heads))
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return x

# create a `Block` class that performs communication then computation (attention --> FFN)
class Block(nn.Module):
    pass
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD) # we represent each token in each batch with just 32 values
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.sa_heads = MultiHead(4, N_EMBD//4)
        self.ffn = FFN(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
    
    def forward(self, index, targets=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = index.shape
        tok_emb = self.table_embedding(index) # (B, T, C), C = N_EMBD
        pos_emb = self.positional_embedding(torch.arange(end=T)) # (T, C), C = N_EMBD -> Broadcast operation applied!
        
        emb = tok_emb + pos_emb # (B, T, C)
        emb = self.sa_heads(emb)
        emb = self.ffn(emb)
        logits = self.lm_head(emb) # (B, T, VOCAB_SIZE)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_tokens_generate):
        # first, get the actual raw values or `logits`
        for _ in range(max_tokens_generate):
            idx_cond = index[:, -BLOCK_SIZE:] # crop to last BLOCK_SIZE tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # this will take just the last timestep for all batches with all vocab size --> (B, C)
            # convert into probabilities from embeddings; we don't want raw logits, we want the converted probabilities or likelihood of what is the next character
            probs = F.softmax(logits, dim=-1) # still (B, C)
            # then sample the next token
            next_idx = torch.multinomial(probs, 1) # then this becomes just (B, 1)
            index = torch.cat((index, next_idx), dim=1)
        return index
    
    def run_train(self, optimizer): # we want to optimize this training loop such that we are averaging out losses over multiple eval cycles. for instance, we can iterate for both training and val, and then get the thing
        for i in range(EPOCHS):
            if i % EVAL_INTERVAL == 0:
                out = evaluate_loss(self)
                print(f"Epoch {i}: train loss (%.4f) {out["train"]:.4f}, val loss (%.4f) {out["val"]:.4f}")
            xb, yb = get_batch("train")
            _, loss = self(xb, yb)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

m = BigramLanguageModel()
optimizer = torch.optim.AdamW(params=m.parameters(), lr=0.001)
m.run_train(optimizer)

idxs = m.generate(torch.zeros((4,1), dtype=torch.int), 100)

for idx in idxs:
    print(decode(idx.tolist()))

