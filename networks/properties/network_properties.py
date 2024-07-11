import numpy as np
import pandas as pd
from alive_progress import alive_bar
import graph_tool.all as gt
import scipy
from plotting import *

#%% CENTRALITY
def calc_centrality(am, nlat, nlon, min_dist=0, max_dist=np.inf):
    # discard edges shorter than min_dist / longer than max_dist
    if min_dist != 0 or max_dist != np.inf:
        dist_matrix = calc_distance_matrix(lat, lat, lon, lon)
        am[dist_matrix < min_dist] = 0
        am[dist_matrix > max_dist] = 0

    # calculate centrality
    out_centrality = np.sum(am, 0) / am.shape[0]
    in_centrality = np.sum(am, 1) / am.shape[0]

    out_centrality_matrix = out_centrality.reshape(nlat, nlon)
    in_centrality_matrix = in_centrality.reshape(nlat, nlon)

    return in_centrality_matrix, out_centrality_matrix

#%% AVERAGE CONNECTIONS
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

#%% LOCAL CLUSTERING
def calc_clustering(am, nlat, nlon):

    g = gt.Graph(scipy.sparse.lil_matrix(am))
    clust = gt.local_clustering(g)
    clust_matrix = clust.get_array().reshape(nlat, nlon)

    return clust_matrix

#%% CLOSENESS CENTRALITY
def calc_closeness(am, nlat, nlon):

    g = gt.Graph(scipy.sparse.lil_matrix(am))
    close = gt.closeness(g)
    close_matrix = close.get_array().reshape(nlat, nlon)

    return close_matrix

#%% BETWEENESS CENTRALITY
def calc_betweeness(am, nlat, nlon):

    g = gt.Graph(scipy.sparse.lil_matrix(am))
    between = gt.closeness(g)
    between_matrix = between.get_array().reshape(nlat, nlon)

    return between_matrix

#%% EIGENVECTOR CENTRALITY
def calc_eigenvector(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    _, eigen = gt.eigenvector(g)
    eigen_matrix = eigen.get_array().reshape(nlat, nlon)

    return eigen_matrix


# %% COMMUNITY DETECTION
def calc_communities(am, nlat, nlon):
    g = gt.Graph(scipy.sparse.lil_matrix(am))
    state = gt.minimize_blockmodel_dl(g)
    com = state.get_blocks()
    del state
    print(np.unique(com))
    com_matrix = com.get_array().reshape(nlat, nlon)

    return com_matrix


#%% PROBABILITY DISTRIBUTION
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

#%% DENSITY
def calc_density(am):
    degrees = np.array([len(np.nonzero(am[i, :])[0]) for i in range(len(am))])
    return degrees

#%% DISTANCE
def calc_distance(lat1, lat2, lon1, lon2):
    R = 6371  # km radius Earth
    Dlat = np.abs(lat1 - lat2)
    Dlon = np.abs(lon1 - lon2)
    a = np.sin(Dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(Dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-np.round(a, 15)))
    return R*c

def calc_distances(am, lon, lat):
    all_dist = np.array([])
    with alive_bar(len(am), force_tty=True) as bar:
        for i in range(len(am)):
            # identify
            lat2 = lat[np.nonzero(am[i, :])[0]]
            lat1 = lat[i]*np.ones_like(lat2)
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