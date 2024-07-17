import numpy as np
def pmethod(m):
    """
    This function calculates the largest eigenvector and eigenvalue of a 2D matrix using power
    method.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type              Description
    ------------------------------------------------------------------------------------------
    m       : numpy.ndarray     Matrix whose largest eigenvec. and eigenval. we want to find.
    ==========================================================================================

    Returns:
    -------
    numpy.ndarray
        largest eigenvector
    int
        largest eigenvalue
    """

    x = np.random.rand(m.shape[0])  # random initial vector
    x = x / np.linalg.norm(x)       # normalise initial vector
    l = np.dot(x.T, np.dot(m, x))   # eigenvalue initial vector

    i = 1
    delta = np.inf
    while delta > 1e-10:

        xx = m @ x
        xx = xx / np.linalg.norm(xx)
        l = xx.T @ m @ xx

        delta = np.linalg.norm(x - xx)
        x = xx

        if i % 1000 == 0:
            print(f"* it {i}: change in eigenvector of {delta}")

        i += 1

    return l, x



