import numpy as np
from pmethod import pmethod

# ----------------------------------------------------------------------------------------------------------------------
# Classes:
# ----------------------------------------------------------------------------------------------------------------------

class AdjacencyMatrix:
    def __init__(self, matrix):
        matrix = matrix                 # adjacency matrix
        k_i = np.sum(matrix, axis=1)    # strength vertices
        m = np.sum(k_i)                 # total strength
class Community:
    def __init__(self, level, indices):
        self.level = level      # Level of the community
        self.indices = indices  # List of vertex indices in this community
        self.left = None        # Left child
        self.right = None       # Right child

# ----------------------------------------------------------------------------------------------------------------------
# Null-models:
# ----------------------------------------------------------------------------------------------------------------------

def configuration_model(A, idx):
    """
        Calculates the modularity matrix of a community based on the configuration null-model.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name        : Type              Description
        ------------------------------------------------------------------------------------------
        A           : AdjacencyMatrix   The adjacency matrix of the overall graph.
        idx         : numpy.ndarray     The indices of the nodes in the community.
        ==========================================================================================

        Returns:
        -------
        numpy.ndarray
            modularity matrix
        """
    kA_i = np.sum(A.matrix[idx, :], axis=1)
    B = (A[idx, idx] - A.k_i[idx].T * A.k_i[idx] / (2*A.m)
         - np.diag(np.diag(kA_i - A.k_i[idx] * np.sum(A.k_i[idx]) / (2*A.m))))
    return B

# ----------------------------------------------------------------------------------------------------------------------
# Modularity optimisation:
# ----------------------------------------------------------------------------------------------------------------------

def find_split(A, idx, modularity):
    """
    Finds the index vector indicating how the community must be split to obtain the optimal
    modularity.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name        : Type              Description
    ------------------------------------------------------------------------------------------
    A           : AdjacencyMatrix   The adjacency matrix of the overall graph.
    idx         : numpy.ndarray     The indices of the nodes in the community.
    ==========================================================================================

    Returns:
    -------
    numpy.ndarray
        index vector
    int
        increase in modularity after split
    """
    B = modularity(A, idx)                              # calculate modularity matrix
    b1 = pmethod(B)                                     # find largest eigenvalue
    s = np.sign(b1)                                     # find communities split
    s[s == 0] = 1 if np.random.rand() < 0.5 else -1     # assign random communities to zero entries
    dQ = s.T @ B @ s                                    # calculate increase in modularity
    return s, dQ

# ----------------------------------------------------------------------------------------------------------------------
# Run function:
# ----------------------------------------------------------------------------------------------------------------------

def split_graph(A, c, modularity=configuration_model):
    """
    Recursively binary-splits communities into sub-communities that maximize modularity using
    a spectral method.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name        : Type              Description
    ------------------------------------------------------------------------------------------
    A           : AdjacencyMatrix   The adjacency matrix of the overall graph.
    c           : Community         The community we want to split.
    modularity  : def               Function calculating the modularity of the null-model we
                                    want to use
    ==========================================================================================
    """

    # find index vector and associated modularity increase
    s, dQ = find_split(A, c.indices, modularity)

    # stop if we cannot split further
    if dQ <= 0:
        return

    # get communities indices
    idx_left, idx_right = np.where(s == 1)[0], np.where(s == -1)[0]

    # create child communities
    c.left = Community(c.level+1, idx_left)
    c.right = Community(c.level+1, idx_right)

    # recursively split the child communities
    split_graph(A, c.left, modularity)
    split_graph(A, c.right, modularity)