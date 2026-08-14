from model.transformer import TransformerLanguageModel

import torch
from torch.nn import functional as F

from config.config import A100ConfigInference

from dataclasses import dataclass

@dataclass
class Request:
    prompt: list[int]
    generated_tokens: list[int]

class Executor:
    def __init__(self, checkpoint_path, device):
        self.device = torch.device(device)
        self.model = TransformerLanguageModel()

        self.checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(
            self.checkpoint['model_save_dict'],
            strict=True
        )

        self.model.to(device=self.device)

        self.model.eval()

        stoi = self.checkpoint["stoi"]
        itos = self.checkpoint["itos"]

        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join([itos[i] for i in l]) 

    @torch.inference_mode()
    def forward(self, requests: list[Request], tokens_to_generate):
        for request in requests:
            request.prompt = request.prompt.to(self.device)
            idx_cond = request.prompt
            for _ in range(tokens_to_generate):
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1]
                out = F.softmax(logits, dim=-1)
                out = torch.multinomial(out, 1)
                idx_cond = torch.cat([idx_cond, out], dim=-1)
            request.generated_tokens = self.decode(idx_cond[0].tolist())
        return requests

config = A100ConfigInference()
executor = Executor("inference/weights/nanogpt.pt", device=config.device)

requests = [
    Request(torch.zeros((1,1), dtype=torch.long), []), 
    Request(torch.zeros((1,1), dtype=torch.long), []), 
    Request(torch.zeros((1,1), dtype=torch.long), [])
]

out = executor.forward(requests, config.block_size - 10)

for r in out:
    print(r.generated_tokens)
    print("**************")