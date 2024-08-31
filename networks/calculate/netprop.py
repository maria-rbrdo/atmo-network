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

from networks.calculate.clustering import *

# ------------------------------------------------------------------------------------------------------------------
# Distance:
# ------------------------------------------------------------------------------------------------------------------
def calc_distance(lat1, lat2, lon1, lon2):
    lat1, lat2, lon1, lon2 = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)
    R = 6371  # km radius Earth
    Dlat = np.abs(lat1 - lat2)
    Dlon = np.abs(lon1 - lon2)
    a = np.sin(Dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(Dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-np.round(a, 15)))
    return R*c

def calc_distances(am, lat, lon):
    all_dist = np.array([])
    with alive_bar(len(am), force_tty=True) as bar:
        for i in range(len(am)):
            # identify
            lat2 = lat[np.nonzero(am[i, :])[0]]
            lat1 = lat[i] * np.ones_like(lat2)
            lon2 = lon[np.nonzero(am[i, :])[0]]
            lon1 = lon[i] * np.ones_like(lon2)
            # calculate
            dist = calc_distance(lat1, lat2, lon1, lon2)
            # append
            all_dist = np.append(all_dist, dist)
            # update bar
            bar()
    return all_dist

def calc_distance_matrix(lat1, lat2, lon1, lon2):
    """
    Calculate the great-circle distance matrix between points specified by latitude and longitude.

    Parameters:
    lat1, lon1: Arrays of latitudes and longitudes for the first set of points.
    lat2, lon2: Arrays of latitudes and longitudes for the second set of points.

    Returns:
    A matrix of distances between each pair of points from the two sets.
    """

    R = 6371  # km radius Earth

    # Create 2D grids of latitudes and longitudes
    lat1_grid, lat2_grid = np.meshgrid(lat1, lat2)
    lon1_grid, lon2_grid = np.meshgrid(lon1, lon2)

    # Calculate differences
    Dlat = lat2_grid - lat1_grid
    Dlon = lon2_grid - lon1_grid

    # Haversine formula
    a = np.sin(Dlat / 2) ** 2 + np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(Dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - np.round(a, 15)))

    return R * c

def calc_avlen(am, nlat, nlon, lat, lon):
    # discard edges shorter than min_dist / longer than max_dist
    dist_matrix = calc_distance_matrix(lat, lat, lon, lon)
    out_avdist = np.sum(am * dist_matrix, 0) / np.sum(am, 0)
    in_avdist = np.sum(am * dist_matrix, 1) / np.sum(am, 1)

    out_avdist_matrix = out_avdist.reshape(nlat, nlon)
    in_avdist_matrix = in_avdist.reshape(nlat, nlon)

    return in_avdist_matrix, out_avdist_matrix

# ----------------------------------------------------------------------------------------------------------------------
# Centrality:
# ----------------------------------------------------------------------------------------------------------------------
def calc_strength(am, nlat, nlon, min_dist=0, max_dist=np.inf, latcorrected=False, lat=None, lon=None):
    # discard edges shorter than min_dist / longer than max_dist
    if min_dist != 0 or max_dist != np.inf:
        dist_matrix = calc_distance_matrix(lat, lat, lon, lon)
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
# Average number of connections:
# ------------------------------------------------------------------------------------------------------------------
def calc_average_connections(am, nlat, nlon, min_dist=0, max_dist=np.inf):
    # discard edges shorter than min_dist / longer than max_dist
    if min_dist != 0 or max_dist != np.inf:
        dist_matrix = calc_distance_matrix(lat, lat, lon, lon)
        am[dist_matrix < min_dist] = 0
        am[dist_matrix > max_dist] = 0

    # calculate average connections
    out_centrality = np.mean(am, 0)
    in_centrality = np.mean(am, 1)

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
def calc_betweeness(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    between = gt.closeness(g)
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


# ------------------------------------------------------------------------------------------------------------------
# Community detection:
# ------------------------------------------------------------------------------------------------------------------
def calc_communities(am, nlat, nlon, lmax=np.inf):
    A = AdjacencyMatrix(am)
    com = np.array(len(am))
    G = Community(level=0, label="0", indices=np.array(range(nlat*nlon)))

    split_graph(A, G, modularity=configuration_model, lmax=lmax)
    assign_communities(G, com, lmax=lmax)

    com_matrix = com.reshape(nlat, nlon)
    return com_matrix

# ----------------------------------------------------------------------------------------------------------------------
# Probability distribution:
# ----------------------------------------------------------------------------------------------------------------------
def calc_cum_prob_distrib(am, measure, savename, dpi=200):
    values, base = np.histogram(am, bins=40)  # evaluate histogram
    cumulative = 1 - np.cumsum(values) / len(am)  # evaluate cumulative
    p = np.polyfit(base[:-1], np.log(cumulative), 1, w=np.sqrt(cumulative))  # fit exponential
    approx = np.exp(p[1]) * np.exp(p[0] * base[:-1])

    # make figure
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    df = pd.DataFrame({'x': base[:-1], 'y': cumulative, 'y_fit': approx})
    sns.lineplot(df, x='x', y='y_fit', ax=ax, markersize=5, linewidth=2, color="gray")
    sns.lineplot(df, x='x', y='y', ax=ax, marker='o', markersize=5, linewidth=2, color="black")
    ax.lines[0].set_linestyle("--")

    ax.set_ylim([1e-3, 1])
    ax.set_yscale('log')

    ax.set_xlabel(f'{measure}')
    ax.set_ylabel('cumulative probability distribution')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    fig.clear()

    return cumulative


# ----------------------------------------------------------------------------------------------------------------------
# Density:
# ----------------------------------------------------------------------------------------------------------------------
def calc_density(am):
    E = np.sum([len(np.nonzero(am[i, :])[0]) for i in range(am.shape[0])])
    N = am.shape[0]
    rho = 2 * E / (N * (N-1))
    return rho
