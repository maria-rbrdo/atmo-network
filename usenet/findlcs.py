import numpy as np
import scipy as sp
from sklearn.cluster import AgglomerativeClustering

def find_split(A, n):
    """
    This function clusters the graph according to simultaneous coherent structure colouring
    (sCSC). It uses agglomerative clustering with complete linkage.
    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type                  Description
    ------------------------------------------------------------------------------------------
    A       : numpy.ndarray         Lagrangian adjacency matrix found using mlagr.py.
    n       : int                   Number of splits
    ==========================================================================================
    """

    D = np.diag(np.sum(A, axis=1))
    L = D - A

    # eigenvalues
    l, X = sp.linalg.eigh(L, D, subset_by_index=[A.shape[0] - n, A.shape[0] - 1])
    X = np.flip(X, axis=1)  # decreasing order

    # split
    S = np.zeros_like(X)
    for i, x in enumerate(X.T):
        clustering = AgglomerativeClustering(linkage="complete").fit(x.reshape(-1, 1))
        S[:, i] = clustering.labels_

    # from binary to int
    s = np.array([''.join(S[i, :].astype('int').astype('str')) for i in range(A.shape[0])])

    return s
