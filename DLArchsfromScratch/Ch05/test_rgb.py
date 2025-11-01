import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Padding helper for channels
# -----------------------------
def pad2d_channels(x, pad):
    # x shape: (C, H, W)
    C, H, W = x.shape
    out = np.zeros((C, H + 2*pad, W + 2*pad))
    out[:, pad:pad+H, pad:pad+W] = x
    return out

# -----------------------------
# Forward pass single observation (C,H,W)
# -----------------------------
def conv2d_forward_obs(x, w):
    # x: (C_in, H, W)
    # w: (C_in, C_out, K, K)
    
    C_in, H, W = x.shape
    C_in2, C_out, K, K2 = w.shape
    assert C_in == C_in2 and K == K2
    
    pad = K // 2
    x_pad = pad2d_channels(x, pad)
    
    out = np.zeros((C_out, H, W))
    
    for c_in in range(C_in):
        for c_out in range(C_out):
            for i in range(H):
                for j in range(W):
                    region = x_pad[c_in, i:i+K, j:j+K]
                    out[c_out, i, j] += np.sum(region * w[c_in, c_out])
    return out, x_pad

# -----------------------------
# Backward pass single observation
# -----------------------------
def conv2d_backward_obs(x, w, x_pad, grad_out):
    # grad_out: (C_out, H, W)
    
    C_in, H, W = x.shape
    C_in2, C_out, K, K2 = w.shape
    assert C_in == C_in2
    
    pad = K // 2
    grad_x_pad = np.zeros_like(x_pad)
    grad_out_pad = pad2d_channels(grad_out, pad)
    
    # dX
    for c_in in range(C_in):
        for c_out in range(C_out):
            for i in range(H):
                for j in range(W):
                    for p in range(K):
                        for q in range(K):
                            grad_x_pad[c_in, i+p, j+q] += \
                                grad_out_pad[c_out, i + (K-p-1), j + (K-q-1)] * w[c_in, c_out, p, q]
    
    grad_x = grad_x_pad[:, pad:-pad, pad:-pad]

    # dW
    grad_w = np.zeros_like(w)
    for c_in in range(C_in):
        for c_out in range(C_out):
            for i in range(H):
                for j in range(W):
                    for p in range(K):
                        for q in range(K):
                            grad_w[c_in, c_out, p, q] += \
                                x_pad[c_in, i+p, j+q] * grad_out[c_out, i, j]

    return grad_x, grad_w

# -----------------------------
# Batch wrapper (N,C,H,W)
# -----------------------------
def conv2d_forward(x, w):
    outs, pads = [], []
    for n in range(x.shape[0]):
        o, p = conv2d_forward_obs(x[n], w)
        outs.append(o)
        pads.append(p)
    return np.stack(outs), pads

def conv2d_backward(x, w, pads, grad_out):
    grad_x_all = []
    grad_w = np.zeros_like(w)
    
    for n in range(x.shape[0]):
        gx, gw = conv2d_backward_obs(x[n], w, pads[n], grad_out[n])
        grad_x_all.append(gx)
        grad_w += gw
    
    return np.stack(grad_x_all), grad_w

# -----------------------------
# Load real RGB image + test
# -----------------------------
img = Image.open("test.jpg").resize((128,128))
x = np.array(img).astype(np.float32) / 255.0   # (H,W,3)
x = x.transpose(2,0,1)                         # -> (C,H,W)
x = np.expand_dims(x, 0)                       # -> (N,C,H,W)

C_in, C_out, K = 3, 3, 3
w = np.random.randn(C_in, C_out, K, K).astype(np.float32) * 0.1

# forward
out, pads = conv2d_forward(x, w)

# backward test
grad_out = np.ones_like(out)
dX, dW = conv2d_backward(x, w, pads, grad_out)

# visualize one channel
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.title("Input"); plt.imshow(x[0].transpose(1,2,0)); plt.axis('off')
plt.subplot(1,2,2); plt.title("Conv Out (ch0)"); plt.imshow(out[0,0], cmap='gnuplot'); plt.axis('off')
plt.show()

print("Output shape:", out.shape)
print("dX shape:", dX.shape)
print("dW shape:", dW.shape)
