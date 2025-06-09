"""
====================================================================================================
Network Properties Script
====================================================================================================
The functions on this script build graphs with the adjacency matrix provided and calculate a network
measure.
----------------------------------------------------------------------------------------------------
"""

import numpy as np
import graph_tool.all as gt
import scipy

def calc_density(am):
    E = np.sum([len(np.nonzero(am[i, :])[0]) for i in range(am.shape[0])])
    N = am.shape[0]
    rho = E / (N * (N - 1))
    return rho

def calc_closeness(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    close = gt.closeness(g)
    close_matrix = close.get_array().reshape(nlat, nlon)

    return close_matrix

def calc_strength(am, nlat, nlon, min_dist=0, max_dist=np.inf, latcorrected=False, lat=None, lon=None):
    # discard edges shorter than min_dist / longer than max_dist
    if min_dist != 0 or max_dist != np.inf:
        dist_matrix = calc_mdist(lat.reshape(-1), lat.reshape(-1), lon.reshape(-1), lon.reshape(-1))
        am[dist_matrix < min_dist] = 0
        am[dist_matrix > max_dist] = 0

    out_centrality = np.sum(am, 0) / am.shape[0]
    in_centrality = np.sum(am, 1) / am.shape[0]

    out_centrality_matrix = out_centrality.reshape(nlat, nlon)
    in_centrality_matrix = in_centrality.reshape(nlat, nlon)

    return in_centrality_matrix, out_centrality_matrix

def calc_clustering(am):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    clust = gt.global_clustering(g, sampled=True)
    return clust

def calc_betweenness(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    between, _ = gt.betweenness(g)
    between_matrix = between.get_array().reshape(nlat, nlon)

    return between_matrix

