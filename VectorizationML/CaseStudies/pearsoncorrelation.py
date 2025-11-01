import numpy as np

def FastPeasonCorrelation(X, Y):
    """
    
    """
    # Zero means
    X_mean = np.mean(X, axis=1, keepdims=True)
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    # normalization
    X_norm = X / np.sqrt(np.sum(X_mean, axis=1, keepdims=True))
    Y_norm = Y / np.sqrt(np.sum(Y_mean, axis=1, keepdims=True))
    Corr = np.sum(X_norm * Y_norm, axis=1, keepdims=True)
    return Corr

if __name__ == "__main__":
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ], dtype=float)
    Y = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ], dtype=float)
    print(FastPeasonCorrelation(X, Y))