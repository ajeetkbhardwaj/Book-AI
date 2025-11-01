import numpy as np

def PairwiseDifference(A, B):
    """
    Compute the pairwise diff between each element of the two
    vectors of length n and m resp and return a matrix of shape
    nxm
    
    * A matrix of shape M x N where the element at (i, j) is A[i]- B[j]
 """
    A = np.asarray(A).flatten()[:, np.newaxis] # shape (M, 1)
    B = np.asarray(B).flatten()[np.newaxis,:] # shape (1, N)
    return A- B # Resulting shape (M, N)

if __name__ == "__main__":
    A = [1, 2, 3]
    B = [4, 5, 6]
    print(PairwiseDifference(A, B))