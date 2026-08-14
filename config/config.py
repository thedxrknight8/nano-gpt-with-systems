from dataclasses import dataclass

@dataclass
class A100ConfigTrain: 
    block_size: int = 256
    batch_size: int = 64
    n_embd: int = 384
    n_heads: int = 6
    n_layers: int = 6
    head_size: int = n_embd // n_heads
    dropout_rate: float = 0.2

    vocab_size: int = 65
    epochs: int = 2500
    learning_rate: float = 3e-4
    eval_interval: int = 300
    eval_iters: int = 200

    device: str = "cuda"

    save_weights: bool = True

@dataclass
class A100ConfigInference: 
    block_size: int = 256
    batch_size: int = 64
    n_embd: int = 384
    n_heads: int = 6
    n_layers: int = 6
    head_size: int = n_embd // n_heads
    dropout_rate: float = 0.2

    vocab_size: int = 65
    epochs: int = 2500
    learning_rate: float = 3e-4
    eval_interval: int = 300
    eval_iters: int = 200

    device: str = "mps"

    save_weights: bool = True