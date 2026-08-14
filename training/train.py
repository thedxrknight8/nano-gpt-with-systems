import torch
from model.transformer import TransformerLanguageModel
from config.config import A100Config

with open("data/input.txt", "r") as file:
    text = file.read()

config = A100Config()
chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

m = TransformerLanguageModel().to(device=config.device)

def get_batch(split, batch_size=config.batch_size):
    data = train if split == "train" else test
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(config.device), y.to(config.device)

data = torch.tensor(encode(text), dtype=torch.long)
train = data[: int(0.9 * len(data))]
test = data[int(0.9 * len(data)) :]

@torch.no_grad()
def evaluate_loss(model):

    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters).to(device=config.device)
        for k in range(config.eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def run_train(model, optimizer):
    for i in range(config.epochs):
        if i % config.eval_interval == 0:
            out = evaluate_loss(model)
            print(
                f'Epoch {i}: train loss (%.4f) {out["train"]:.4f}, val loss (%.4f) {out["val"]:.4f}'
            )
        xb, yb = get_batch("train")

        _, loss = m.forward(xb, yb)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

optimizer = torch.optim.AdamW(params=m.parameters(), lr=config.learning_rate)
run_train(m, optimizer)

if config.save_weights:
    torch.save({
        "model_save_dict": m.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {
            "BLOCK_SIZE": config.block_size,
            "BATCH_SIZE": config.batch_size,
            "EPOCHS": config.epochs,
            "LEARNING_RATE": config.learning_rate,
            "N_EMBD": config.n_embd,
            "N_HEADS": config.n_heads,
            "N_LAYERS": config.n_layers
        }
    }, "/outputs/nanogpt.pt")