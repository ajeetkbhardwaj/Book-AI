import numpy as np

def pad2d(x, pad):
    return np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))

def conv2d_forward(x, w, pad=1):
    # x: (B, Cin, H, W)
    # w: (Cout, Cin, k, k)
    B, Cin, H, W = x.shape
    Cout, _, k, _ = w.shape
    
    x_pad = pad2d(x, pad)
    out = np.zeros((B, Cout, H, W))
    
    for b in range(B):
        for co in range(Cout):
            for i in range(H):
                for j in range(W):
                    region = x_pad[b, :, i:i+k, j:j+k]
                    out[b, co, i, j] = np.sum(region * w[co])
    return out

def conv2d_backward(x, w, grad_out, pad=1):
    B, Cin, H, W = x.shape
    Cout, _, k, _ = w.shape
    
    x_pad = pad2d(x, pad)
    grad_out_pad = pad2d(grad_out, pad)
    
    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)
    grad_x_pad = np.zeros_like(x_pad)
    
    # grad wrt filters
    for b in range(B):
        for co in range(Cout):
            for ci in range(Cin):
                for i in range(H):
                    for j in range(W):
                        region = x_pad[b, ci, i:i+k, j:j+k]
                        grad_w[co, ci] += region * grad_out[b, co, i, j]
    
    # grad wrt input
    for b in range(B):
        for co in range(Cout):
            for i in range(H):
                for j in range(W):
                    region = w[co]
                    grad_x_pad[b, :, i:i+k, j:j+k] += region * grad_out[b, co, i, j]
    
    grad_x = grad_x_pad[:, :, pad:-pad, pad:-pad]
    return grad_x, grad_w

x = np.random.randn(2, 3, 5, 5)   # B=2, Cin=3, H=W=5
w = np.random.randn(4, 3, 3, 3)   # Cout=4, Cin=3, k=3

y = conv2d_forward(x, w, pad=1)
grad_out = np.random.randn(*y.shape)
grad_x, grad_w = conv2d_backward(x, w, grad_out, pad=1)

print(y.shape)          # (2,4,5,5)
print(grad_x.shape)     # (2,3,5,5)
print(grad_w.shape)     # (4,3,3,3)
