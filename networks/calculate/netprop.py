"""
========================================================================================================================
Network Properties Script
========================================================================================================================
The functions on this script build graphs with the adjacency matrix provided and calculate a network measure.
"""

import numpy as np
import pandas as pd
from alive_progress import alive_bar
import graph_tool.all as gt
import scipy

# ------------------------------------------------------------------------------------------------------------------
# Distance:
# ------------------------------------------------------------------------------------------------------------------
def calc_mdist(lat1, lat2, lon1, lon2):
    """
    Calculate the great-circle distance matrix between points specified by latitude and longitude.

    Parameters:
    lat1, lon1: Arrays of latitudes and longitudes for the first set of points.
    lat2, lon2: Arrays of latitudes and longitudes for the second set of points.

    Returns:
    A matrix of distances between each pair of points from the two sets.
    """

    R = 6371000  # km radius Earth

    lat1, lat2, lon1, lon2 = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)

    # Create 2D grids of latitudes and longitudes
    lat1_grid, lat2_grid = np.meshgrid(lat1, lat2)
    lon1_grid, lon2_grid = np.meshgrid(lon1, lon2)

    # Calculate differences
    Dlat = lat2_grid - lat1_grid
    Dlon = lon2_grid - lon1_grid

    # Haversine formula
    a = np.sin(Dlat / 2) ** 2 + np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(Dlon / 2) ** 2
    c = np.arctan2(np.sqrt(a), np.sqrt(1 - np.round(a, 15)))

    return 2 * R * c

def calc_distance(am, nlat, nlon, lat, lon):

    dist_matrix = calc_mdist(lat, lat, lon, lon)

    sum0, sum1 = np.sum(am, 0), np.sum(am, 1)
    sum0[sum0 == 0] = np.inf
    sum1[sum1 == 0] = np.inf

    out_dist = np.sum(am * dist_matrix, 0) / np.sum(am, 0) * np.sin(np.deg2rad(lat)).reshape(-1)
    in_dist = np.sum(am * dist_matrix, 1) / np.sum(am, 1) * np.sin(np.deg2rad(lat)).reshape(-1)

    out_dist_matrix = out_dist.reshape(nlat, nlon)
    in_dist_matrix = in_dist.reshape(nlat, nlon)

    return in_dist_matrix, out_dist_matrix

# ----------------------------------------------------------------------------------------------------------------------
# Centrality:
# ----------------------------------------------------------------------------------------------------------------------
def calc_strength(am, nlat, nlon, min_dist=0, max_dist=np.inf, latcorrected=False, lat=None, lon=None):
    # discard edges shorter than min_dist / longer than max_dist
    if min_dist != 0 or max_dist != np.inf:
        dist_matrix = calc_mdist(lat, lat, lon, lon)
        am[dist_matrix < min_dist] = 0
        am[dist_matrix > max_dist] = 0

    # calculate centrality
    if latcorrected:
        out_centrality = np.sum(am * np.cos(np.deg2rad(lat)).reshape(-1, 1), 0) / am.shape[0]
        in_centrality = np.sum(am * np.cos(np.deg2rad(lat)).reshape(-1, 1), 1) / am.shape[0]
    else:
        out_centrality = np.sum(am, 0) / am.shape[0]
        in_centrality = np.sum(am, 1) / am.shape[0]

    out_centrality_matrix = out_centrality.reshape(nlat, nlon)
    in_centrality_matrix = in_centrality.reshape(nlat, nlon)

    return in_centrality_matrix, out_centrality_matrix

# ------------------------------------------------------------------------------------------------------------------
# Global clustering:
# ------------------------------------------------------------------------------------------------------------------
def calc_clustering(am):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    clust = gt.global_clustering(g, sampled=True)
    return clust

# ------------------------------------------------------------------------------------------------------------------
# Closeness:
# ------------------------------------------------------------------------------------------------------------------
def calc_closeness(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    close = gt.closeness(g)
    close_matrix = close.get_array().reshape(nlat, nlon)

    return close_matrix

# ----------------------------------------------------------------------------------------------------------------------
# Betweeness:
# ----------------------------------------------------------------------------------------------------------------------
def calc_betweenness(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    between, _ = gt.betweenness(g)
    between_matrix = between.get_array().reshape(nlat, nlon)

    return between_matrix

# ----------------------------------------------------------------------------------------------------------------------
# Eigenvector centrality:
# ----------------------------------------------------------------------------------------------------------------------
def calc_eigenvector(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    _, eigen = gt.eigenvector(g)
    eigen_matrix = eigen.get_array().reshape(nlat, nlon)

    return eigen_matrix

# ----------------------------------------------------------------------------------------------------------------------
# Density:
# ----------------------------------------------------------------------------------------------------------------------
def calc_density(am):
    E = np.sum([len(np.nonzero(am[i, :])[0]) for i in range(am.shape[0])])
    N = am.shape[0]
    rho = E / (N * (N - 1))
    return rho

