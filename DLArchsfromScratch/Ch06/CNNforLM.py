import numpy as np

# ---------- Text ----------
text = "Lello convolution RDXF"
vowels = set("aeiou")

y = np.array([1 if c in vowels else 0 for c in text])  # vowel=1 consonant=0

chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}

embedding_dim = 4
np.random.seed(42)

# parameters
E = np.random.randn(len(chars), embedding_dim) * 0.1
w = np.random.randn(3, embedding_dim) * 0.1
W_out = np.random.randn(embedding_dim, 2) * 0.1  # classifier weights
b_out = np.zeros(2)

# ---------- Utilities ----------
def pad1d(x, pad):
    return np.pad(x, ((pad, pad), (0,0)), mode='constant')

def conv1d_forward(x, w):
    T, D = x.shape
    k = w.shape[0]
    pad = k // 2
    x_pad = pad1d(x, pad)
    out = np.zeros((T, D))
    for i in range(T):
        for p in range(k):
            out[i] += x_pad[i+p] * w[p]
    return out, x_pad

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

# ---------- Forward ----------
x_idx = np.array([stoi[c] for c in text])
x = E[x_idx]

conv_out, _ = conv1d_forward(x, w)

h = conv_out.mean(axis=0)     # mean pooling
logits = h @ W_out + b_out
probs = softmax(logits)
pred = np.argmax(probs)

print("Text:", text)
print("Predicted class:", pred, "(0=consonant-seq, 1=vowel-seq)")
print("Probabilities:", probs)
