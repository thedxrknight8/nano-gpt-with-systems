# nano-gpt-with-systems

An educational, from-scratch implementation of a character-level GPT in
PyTorch. The project builds up from a bigram language model to a decoder-only
transformer and includes a simple path for running the larger model on a cloud
GPU.

The goal is to keep the full training and generation pipeline small enough to
read in one sitting while exposing the systems details that affect language
model execution: batching, device placement, autoregressive generation, and
latency measurement.

## What It Implements

- Character-level tokenization over a Shakespeare corpus
- Random next-token training batches
- Learned token and positional embeddings
- Masked, scaled dot-product self-attention
- Multi-head attention
- Feed-forward networks
- Pre-normalization and residual connections
- Dropout
- Autoregressive sampling
- Automatic CUDA, Apple Silicon, or CPU device selection
- Time to first token (TTFT) and average time between tokens (TBT)

## Model Progression

| File                | Stage                    | Main additions                                                                                                                |
| ------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `scripts/bigram.py` | Bigram baseline          | Token-to-next-token lookup table and cross-entropy training                                                                   |
| `scripts/v2.py`     | First attention model    | Token positions, causal self-attention, multiple heads, and a feed-forward layer                                              |
| `scripts/v3.py`     | Decoder-only transformer | Six transformer blocks, residual connections, layer normalization, dropout, device placement, generation, and latency metrics |

`v3.py` is the current documented baseline.

## Architecture

The v3 model operates on tensors of token IDs with shape `(B, T)`, where `B`
is the batch size and `T` is the sequence length.

```text
token IDs
   |
   +-- token embeddings --------+
   |                            |
   +-- positional embeddings ---+
                                |
                      6 x Transformer Block
                      +---------------------+
                      | LayerNorm           |
                      | Multi-Head Attention|
                      | Residual connection |
                      | LayerNorm           |
                      | Feed-Forward Network|
                      | Residual connection |
                      +---------------------+
                                |
                            LM head
                                |
                     next-character logits
```

Each attention head computes:

```text
Attention(Q, K, V) = softmax((QK^T / sqrt(head_size)) + causal_mask) V
```

The lower-triangular causal mask prevents a token from attending to future
positions. The resulting logits are trained against the input shifted by one
character using cross-entropy loss.

### v3 Configuration

| Parameter          |      Value |
| ------------------ | ---------: |
| Context length     | 256 tokens |
| Batch size         |         64 |
| Embedding width    |        384 |
| Attention heads    |          6 |
| Head width         |         64 |
| Transformer layers |          6 |
| Dropout            |        0.2 |
| Learning rate      |     `3e-4` |
| Training steps     |      2,500 |
| Generated tokens   |        250 |

All hyperparameters are defined near the top of `scripts/v3.py`.

## Repository Layout

```text
.
├── data/
│   └── input.txt          # Character-level Shakespeare training corpus
├── gpu-run/
│   └── modal_run.py       # Modal A10G training launcher
├── metrics/
│   └── metrics.txt        # Saved generation latency measurements
├── notebooks/
│   └── nano-gpt.ipynb     # Experimentation notebook
└── scripts/
    ├── bigram.py          # Bigram baseline
    ├── v2.py              # Initial attention implementation
    └── v3.py              # Full decoder-only transformer baseline
```

## Setup

Create a virtual environment and install PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

The scripts expect to be launched from the repository root so that
`data/input.txt` resolves correctly.

## Running Locally

Run the smaller attention model:

```bash
python scripts/v2.py
```

Run the v3 transformer:

```bash
python scripts/v3.py
```

The script selects the best available device in this order:

1. NVIDIA CUDA
2. Apple Metal Performance Shaders (MPS)
3. CPU

Training starts immediately, periodically reports average training and
validation loss, generates text after the final step, and then records
generation latency. The current v3 script writes those metrics to
`/outputs/metrics.txt`, which is mounted automatically by the Modal launcher.
A local run needs that path to be writable.

## Running on Modal

The included Modal definition builds a minimal image with PyTorch, uploads the
repository, attaches a persistent output volume, and runs v3 on an NVIDIA A10G:

```bash
python -m pip install modal
modal setup
modal run gpu-run/modal_run.py::train
```

The remote function has a one-hour timeout. Metrics written under `/outputs`
are persisted in the `nano-gpt-runs` Modal volume.

## Metrics

Generation records:

- **TTFT**: elapsed time from the start of generation until the first sampled
  token is available.
- **TBT**: average elapsed time between subsequent sampled tokens.

The checked-in measurement for a 250-token generation is:

| Metric      |    Value |
| ----------- | -------: |
| TTFT        | 84.42 ms |
| Average TBT | 10.53 ms |

These numbers are a single saved run, not a controlled benchmark. Hardware,
PyTorch version, model state, warmup, and synchronization behavior all affect
the result.

## Current Scope

This repository intentionally favors direct, readable scripts over framework
abstractions. At the current stage:

- Training begins when a script is executed; there is no command-line
  configuration layer.
- Model checkpoints are not saved or restored.
- Tokenization is character-level and tied to the supplied corpus vocabulary.
- Sampling uses multinomial sampling without temperature, top-k, or top-p
  controls.
- The scripts do not yet include an automated test suite.

These constraints keep the implementation compact and make each stage of the
model easy to inspect.
