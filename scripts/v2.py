import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
BLOCK_SIZE = 8
BATCH_SIZE = 4
EPOCHS = 10000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
VOCAB_SIZE = 65
EVAL_ITERS = 200
N_EMBD = 32
# ---------------

torch.manual_seed(1337)

with open("../data/input.txt", "r") as file: 
    text = file.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# conversion sets for characters from token indexes
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# splitting the data in `train` and `test`
data = torch.tensor(encode(text), dtype=torch.long)
train = data[:int(0.9 * len(data))]
test = data[int(0.9 * len(data)) + 1:]

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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD) # we represent each token in each batch with just 32 values
        self.positional_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
    
    def forward(self, index, targets=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = index.shape
        tok_emb = self.table_embedding(index) # (B, T, C), C = N_EMBD
        pos_emb = self.positional_embedding(torch.arange(end=T)) # (T, C), C = N_EMBD -> Broadcast operation applied!
        
        emb = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(emb) # (B, T, VOCAB_SIZE)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_tokens_generate):
        # first, get the actual raw values or `logits`
        for _ in range(max_tokens_generate):
            logits, _ = self(index)
            logits = logits[:, -1, :] # this will take just the last timestep for all batches with all vocab size --> (B, C)
            # convert into probabilities from embeddings; we don't want raw logits, we want the converted probabilities or likelihood of what is the next character
            probs = F.softmax(logits, dim=-1) # still (B, C)
            # then sample the next token
            next_idx = torch.multinomial(probs, 1) # then this becomes just (B, 1)
            index = torch.cat((index, next_idx), dim=1)
        return index
    
    def run_train(self, optimizer): # we want to optimize this training loop such that we are averaging out losses over multiple eval cycles. for instance, we can iterate for both training and val, and then get the thing
        for i in range(EPOCHS):
            if i % EVAL_ITERS == 0:
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

print(m.generate(torch.zeros((4,0,0), dtype=torch.int), 100))

