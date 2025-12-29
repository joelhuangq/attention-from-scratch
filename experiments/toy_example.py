import numpy as np
from src.attention_numpy import scaled_dot_product_attention

np.random.seed(42)

n, d = 3, 2
X = np.random.randn(n, d)

W_Q = np.eye(d)
W_K = np.eye(d)
W_V = np.eye(d)

output = scaled_dot_product_attention(X, W_Q, W_K, W_V)

print("Input X:\n", X)
print("Attention Output:\n", output)
