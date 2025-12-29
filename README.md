# Attention from Scratch

This repository records my step-by-step understanding and implementation of
the Attention mechanism used in Transformer-based large language models.

The goal is to start from **pure linear algebra**, avoid framework abstractions,
and gradually build toward practical implementations.

---

## What I Learned

- Attention is a **learned weighted averaging operator**
- Query, Key, Value are linear projections of the same input space
- Softmax turns similarity scores into a differentiable attention distribution
- Multi-Head Attention enables parallel attention subspaces

---

## Mathematical Formulation

Given input matrix:

X ∈ ℝ^{n × d}

Linear projections:

Q = X W_Q  
K = X W_K  
V = X W_V  

Scaled dot-product attention:

Attention(X) = softmax( Q Kᵀ / √d ) V

---

## Repository Structure

```text
.
├── README.md
├── notes/
│   └── attention_linear_algebra.md
├── src/
│   ├── attention_numpy.py
│   └── softmax.py
└── experiments/
    └── toy_example.py
