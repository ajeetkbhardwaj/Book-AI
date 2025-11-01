import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# Utility: pad 2D
# ----------------------------
def pad2d(x, pad):
    return np.pad(x, ((pad,pad), (pad,pad)), mode='constant')

# ----------------------------
# Conv2D forward
# x: HxW, w: kxk
# ----------------------------
def conv2d_forward(x, w):
    H, W = x.shape
    k = w.shape[0]
    pad = k // 2

    x_pad = pad2d(x, pad)
    out = np.zeros_like(x)

    for i in range(H):
        for j in range(W):
            region = x_pad[i:i+k, j:j+k]
            out[i,j] = np.sum(region * w)
    return out, x_pad

# ----------------------------
# Conv2D backward
# grad_out: HxW
# returns dX, dW
# ----------------------------
def conv2d_backward(x, w, x_pad, grad_out):
    H, W = x.shape
    k = w.shape[0]
    pad = k // 2

    # dX
    grad_x_pad = np.zeros_like(x_pad)
    grad_out_pad = pad2d(grad_out, pad)

    for i in range(H):
        for j in range(W):
            for p in range(k):
                for q in range(k):
                    grad_x_pad[i+p, j+q] += grad_out_pad[i + (k-p-1), j + (k-q-1)] * w[p,q]

    grad_x = grad_x_pad[pad:-pad, pad:-pad]

    # dW
    grad_w = np.zeros_like(w)
    for i in range(H):
        for j in range(W):
            for p in range(k):
                for q in range(k):
                    grad_w[p,q] += x_pad[i+p, j+q] * grad_out[i,j]

    return grad_x, grad_w

# ----------------------------
# Test on real image
# ----------------------------
img = Image.open("test.jpg").convert("L")
img = img.resize((128,128))
x = np.array(img, dtype=np.float32) / 255.0

# example filter (edge detector)
w = np.array([[ -1, -1, -1],
              [ -1,  8, -1],
              [ -1, -1, -1 ]], dtype=np.float32)

# forward
out, x_pad = conv2d_forward(x, w)

# backward test: use ones as grad_out
grad_out = np.ones_like(out)
dX, dW = conv2d_backward(x, w, x_pad, grad_out)

# visualize forward
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.title("Input"); plt.imshow(x, cmap='gray'); plt.axis("off")
plt.subplot(1,2,2); plt.title("Conv Output"); plt.imshow(out, cmap='gray'); plt.axis("off")
plt.show()

print("dX shape:", dX.shape)
print("dW:\n", dW)
