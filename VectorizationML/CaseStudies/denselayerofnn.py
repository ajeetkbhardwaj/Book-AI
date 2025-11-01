import numpy as np

# activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
# linear opeartion
def linear(x, w, b):
    return x @ w + b

# feed forward dense layer
def ff_layer(x, w, b):
    return sigmoid(linear(x, w, b))

if __name__ == "__main__":
    x = np.array([0,1,2,3,4,5,6,7,8,9], dtype=float)  # input
    w = np.array([0.5], dtype=float)                 # weight
    b = np.array([1.0], dtype=float)                 # bias

    z = ff_layer(x, w, b)
    print(z)

    # input matrix (4 samples, 3 features)
    X = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 3, 4],
        [4, 5, 6]
    ], dtype=float)

    # weight matrix (3 input features -> 2 neurons)
    W = np.array([
        [0.2, 0.1],
        [0.3, 0.4],
        [0.5, 0.6]
    ], dtype=float)

    # bias vector for 2 neurons
    b1 = np.array([0.1, -0.2], dtype=float)

    Z = ff_layer(X, W, b1)
    print(Z)