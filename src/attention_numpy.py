import numpy as np
from softmax import softmax

def scaled_dot_product_attention(X, W_Q, W_K, W_V):
    """
    X: (n, d)
    W_Q, W_K, W_V: (d, d)
    """
    d = X.shape[1]

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T / np.sqrt(d)
    A = softmax(scores, axis=-1)

    return A @ V
