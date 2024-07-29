import numpy as np
from networks.calculate.pmethod import pmethod

# ----------------------------------------------------------------------------------------------------------------------
# Classes:
# ----------------------------------------------------------------------------------------------------------------------

class AdjacencyMatrix:
    def __init__(self, matrix):
        matrix = matrix                 # adjacency matrix
        k_i = np.sum(matrix, axis=1)    # strength vertices
        m = np.sum(k_i)                 # total strength
class Community:
    def __init__(self, level, label, indices):
        self.level = level      # Level of the community
        self.label = label      # Label of the community
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

def split_graph(A, c, modularity=configuration_model, lmax=np.inf):
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
    lmax        : int               The maximum level in the graph we want to calculate. If
                                    lmax = np.inf it continues until communities are indivisible.
    ==========================================================================================
    """

    # find index vector and associated modularity increase
    s, dQ = find_split(A, c.indices, modularity)

    # stop if we cannot split further
    if dQ <= 0 or c.level + 1 > lmax:
        return

    # get communities indices
    idx_left, idx_right = np.where(s == 1)[0], np.where(s == -1)[0]

    # create child communities
    c.left = Community(c.level+1, c.label+"1", idx_left)    # left splits are indicated with 1
    c.right = Community(c.level+1, c.label+"0", idx_right)  # right splits are indicated with 0

    # recursively split the child communities
    split_graph(A, c.left, modularity, lmax)
    split_graph(A, c.right, modularity, lmax)

# ----------------------------------------------------------------------------------------------------------------------
# Assign communities:
# ----------------------------------------------------------------------------------------------------------------------

def assign_communities(c, communities, lmax=np.inf):
    """
    Assign community to each node in the graph.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name        : Type              Description
    ------------------------------------------------------------------------------------------
    c           : Community         The communities we want to assign.
    communities : numpy.ndarray     The array with all the nodes in the graph.
    lmax        : int               The maximum level in the graph we want to see.
    ==========================================================================================
    """

    # stop if this is the last community
    if c is None or c.level > lmax:
        return

    # assign the current label to all indices in this community
    for index in c.indices:
        communities[index] = c.label

    # Recursively assign labels to child communities
    assign_communities(c.left, communities)
    assign_communities(c.right, communities)
