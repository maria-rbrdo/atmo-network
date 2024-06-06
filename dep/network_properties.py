import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from graph_tool.all import *
import scipy

#%% PLOT
def plot_matrix(matrix, measure, lon, lat, times, savename, dpi=200, vmax=None, vmin=None):
    #make figure
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})
    sns.heatmap(np.flip(matrix.T, 0), ax=ax, center = 0,
                #cmap=sns.cubehelix_palette(as_cmap=True, start=.5, rot=-.75, reverse=True),
                cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True),
                cbar_kws=dict(use_gridspec=False, location="top", aspect=60, extend='both',
                              label=f"{measure}", pad=0.01))
    x_ticks = 9
    y_ticks = 5
    ax.set_xticks(np.linspace(0, len(np.unique(lon)), x_ticks))
    ax.set_xticklabels(np.linspace(-180, 180, x_ticks, dtype=int))
    ax.set_yticks(np.linspace(0, len(np.unique(lat)), y_ticks))
    ax.set_yticklabels(np.linspace(90, -90, y_ticks, dtype=int))

    ax.set_xlabel('longitude (deg)')
    ax.set_ylabel('latitude (deg)')

    fig.suptitle('t = {:.3f} - {:.3f} hrs'.format(times[0] / 1000, times[1] / 1000))

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    fig.clear()

#%% CENTRALITY
def calc_centrality(cm, lon, lat, times, savename, dpi=200, area_weighted=False):
    # calculate centrality
    if area_weighted is False:
        centrality = np.sum(cm, 1) / cm.shape[0]
    else:
        centrality = np.sum(cm, 1)*np.cos(np.deg2rad(lat)) / np.sum(np.cos(np.deg2rad(lat)))

    centrality_matrix = centrality.reshape(-1, (lon == 0).sum())

    # generate plot
    plot_matrix(centrality_matrix, "normalised strength centrality", lon, lat, times, savename, dpi=dpi)

    return centrality

#%% LOCAL CLUSTERING
def calc_clustering(cm, lon, lat, times, savename, dpi=200):

    g = Graph(scipy.sparse.lil_matrix(cm))
    clust = local_clustering(g)
    clust_matrix = clust.get_array().reshape(-1, (lon == 0).sum())

    # generate plot
    plot_matrix(clust_matrix, "local clustering", lon, lat, times, savename, dpi=dpi)

    return clust.get_array()

#%% CLOSENESS CENTRALITY
def calc_closeness(cm, lon, lat, times, savename, dpi=200):

    g = Graph(scipy.sparse.lil_matrix(cm))
    close = closeness(g)
    close_matrix = close.get_array().reshape(-1, (lon == 0).sum())

    # generate plot
    plot_matrix(close_matrix, "closeness centrality", lon, lat, times, savename, dpi=dpi)

    return close.get_array()

#%% BETWEENESS CENTRALITY
def calc_betweeness(cm, lon, lat, times, savename, dpi=200):

    g = Graph(scipy.sparse.lil_matrix(cm))
    between = closeness(g)
    between_matrix = between.get_array().reshape(-1, (lon == 0).sum())

    # generate plot
    plot_matrix(between_matrix, "betweeness centrality", lon, lat, times, savename, dpi=dpi)

    return between.get_array()


#%% EIGENVECTOR CENTRALITY
def calc_eigenvector(cm, lon, lat, times, savename, dpi=200):
    g = Graph(scipy.sparse.lil_matrix(cm))
    _, eigen = eigenvector(g)
    eigen_matrix = eigen.get_array().reshape(-1, (lon == 0).sum())

    # generate plot
    plot_matrix(eigen_matrix, "eigenvector centrality", lon, lat, times, savename, dpi=dpi)

    return eigen.get_array()

#%% PROBABILITY DISTRIBUTION
def calc_prob_distrib(matrix, measure, savename, dpi=200):
    values, base = np.histogram(matrix, bins=40)  # evaluate histogram
    cumulative = 1 - np.cumsum(values) / len(matrix)  # evaluate cumulative
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
def calc_density(matrix):
    degrees = np.array([len(np.nonzero(matrix[i, :])[0]) for i in range(len(matrix))])
    return degrees

#%% DISTANCE
def calc_distance(lat1, lat2, lon1, lon2):
    R = 6371  # km radius Earth
    Dlat = np.abs(lat1 - lat2)
    Dlon = np.abs(lon1 - lon2)
    a = np.sin(Dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(Dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-np.round(a, 15)))
    return R*c
def calc_distances(matrix, lon, lat):
    all_dist = np.array([])
    with alive_bar(len(matrix), force_tty=True) as bar:
        for i in range(len(matrix)):
            # identify
            lat2 = lat[np.nonzero(matrix[i, :])[0]]
            lat1 = lat[i]*np.ones_like(lat2)
            lon2 = lon[np.nonzero(matrix[i, :])[0]]
            lon1 = lon[i] * np.ones_like(lon2)
            # calculate
            dist = calc_distance(lat1, lat2, lon1, lon2)
            # append
            all_dist = np.append(all_dist, dist)
            # update bar
            bar()
    return all_dist
