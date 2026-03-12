# PyTorch Neural Network Learning Repo

This repository is a hands-on workspace for learning PyTorch, CUDA basics, and the main neural network architectures step by step.

The focus is practical:

- understand how PyTorch tensors and training loops work
- use your NVIDIA GPU through CUDA
- learn when to use MLPs, CNNs, RNNs, LSTMs, and Transformers
- build small runnable examples instead of only reading theory

## Prerequisites

This learning guide assumes you already have some basic background in:

- machine learning fundamentals
- basic NLP concepts

Examples of useful prior knowledge:

- features, labels, training, validation, and testing
- classification and regression
- overfitting and generalization
- tokenization, embeddings, and sequence data at a beginner level

This repository is focused on neural networks with PyTorch, so it is meant to build on top of that foundation rather than replace it.

## Current focus

Right now this repo includes:

- a CUDA check script to confirm PyTorch can see the GPU
- a small CUDA basics demo using PyTorch tensors
- a learning folder scaffold organized by model family

## Project structure

- `test.py`: verifies CUDA availability and shows the detected GPU
- `cuda_basics.py`: simple GPU tensor math, matrix multiplication, and autograd examples
- `learning/`: guided structure for each neural network family

Inside `learning/`:

- `mlp/`: feedforward multilayer perceptrons
- `cnn/`: convolutional neural networks
- `rnn/`: basic recurrent neural networks
- `lstm/`: long short-term memory networks
- `transformers/`: attention-based sequence models

## Learning path

Recommended order:

1. MLP
2. CNN
3. RNN
4. LSTM
5. Transformer

This order matters because each step adds one major idea:

- `MLP`: the basic training loop, layers, activations, loss, and optimization
- `CNN`: spatial structure for image-like inputs
- `RNN`: sequential processing over time
- `LSTM`: better memory for longer sequences
- `Transformer`: attention-based modeling used in most modern language systems

## Run the current examples

```powershell
uv run test.py
uv run cuda_basics.py
```

## What `cuda_basics.py` demonstrates

- creating tensors directly on the GPU
- running parallel tensor operations on CUDA
- timing a large matrix multiplication
- computing gradients with autograd on CUDA tensors

## What comes next

The next step is to add content progressively to each folder in `learning/`, starting with `mlp/`.

Each model family will eventually include:

- a short concept explanation
- expected input and output shapes
- a minimal training script
- a small dataset example
- notes on when to use that architecture
