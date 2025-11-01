import numpy as np

# Padding helper
def pad_1d(inp, pad):
    return np.concatenate([np.zeros(pad), inp, np.zeros(pad)])

# Forward 1D convolution (stride = 1)
def conv1d_forward(inp, kernel):
    k = kernel.shape[0]
    pad = k // 2
    x_pad = pad_1d(inp, pad)
    out = np.zeros_like(inp, dtype=float)
    
    for i in range(len(inp)):
        for j in range(k):
            out[i] += kernel[j] * x_pad[i + j]
    return out

# Backward pass: compute gradients wrt input and kernel
def conv1d_backward(inp, kernel, grad_out):
    k = kernel.shape[0]
    pad = k // 2
    x_pad = pad_1d(inp, pad)
    grad_out_pad = pad_1d(grad_out, pad)

    grad_inp = np.zeros_like(inp, dtype=float)
    grad_kernel = np.zeros_like(kernel, dtype=float)

    # input gradient
    for i in range(len(inp)):
        for j in range(k):
            grad_inp[i] += grad_out_pad[i + k - j - 1] * kernel[j]

    # kernel gradient
    for i in range(len(inp)):
        for j in range(k):
            grad_kernel[j] += x_pad[i + j] * grad_out[i]

    return grad_inp, grad_kernel

# ----- Test -----

inp = np.array([1, 2, 3, 4, 5], dtype=float)
kernel = np.array([0, 1, 0], dtype=float)

# Forward()
out = conv1d_forward(inp, kernel)
print("Forward Output:", out)

# Loss = sum(out) → grad_out = 1 for each output
grad_out = np.ones_like(out)

g_inp, g_kernel = conv1d_backward(inp, kernel, grad_out)
print("Gradient wrt input:", g_inp)
print("Gradient wrt kernel:", g_kernel)

# Gradient check: perturb input element
epsilon = 1.0
inp2 = inp.copy()
inp2[4] += epsilon   # increase last element by 1

loss1 = np.sum(conv1d_forward(inp, kernel))
loss2 = np.sum(conv1d_forward(inp2, kernel))

print("Loss diff =", loss2 - loss1, "| Analytical grad =", g_inp[4])
