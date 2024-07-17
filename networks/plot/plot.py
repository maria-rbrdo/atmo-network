import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import cartopy.crs as ccrs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Heat Map:
# ----------------------------------------------------------------------------------------------------------------------

def plot_matrix(ax, matrix, lat, lon, my_cmap=None, min=None, max=None, levels=25, H=None):
    # define topography
    if H is not None:
        A0 = 0.15 * H
        hb = lambda lon, lat: A0 * np.sin(2 * lat) ** 2 * np.cos(2 * lon)

    # select colormap depending on whether all values have the same sign or not
    if my_cmap is None:
        if np.all(matrix >= 0):
            my_cmap = sns.cm.rocket
        elif np.all(matrix <= 0):
            my_cmap = sns.cm.mako
            matrix = abs(matrix)
        else:
            my_cmap = sns.color_palette("icefire", as_cmap=True)

    # make figure
    x, y, z = lon, lat, matrix

    x = np.concatenate((np.atleast_1d(0), x, np.atleast_1d(360)))
    y = np.concatenate((np.atleast_1d(90), y, np.atleast_1d(-90)))
    z = np.concatenate((np.atleast_2d([np.mean(z[0, :])]*len(z[0, :])), z,
                        np.atleast_2d([np.mean(z[-1, :])]*len(z[-1, :]))), axis=0)
    z = np.concatenate((np.atleast_2d(z[:, 0]).T, z, np.atleast_2d(z[:, -1]).T), axis=1)

    ax.contourf(x, y, z, levels, transform=ccrs.PlateCarree(), cmap=my_cmap, vmin=min, vmax=max)

    if H is not None:
        X, Y = np.meshgrid(np.deg2rad(x), np.deg2rad(y))
        HB = hb(X, Y)
        ax.contour(x, y, HB, colors='white', linewidths=3, transform=ccrs.PlateCarree())

# ----------------------------------------------------------------------------------------------------------------------
# Cumulative probability distribution:
# ----------------------------------------------------------------------------------------------------------------------

def plot_cumsum(ax, data_lst, names_lst, lmax, ptype, colors=[None,None]):
    for i in range(len(data_lst)):
        values, base = np.histogram(data_lst[i].reshape(-1), bins=5000)
        cumulative = np.cumsum(values)/len(data_lst[i].reshape(-1))
        ax.plot(base[:-1], 1-cumulative, "-", linewidth=5, markersize=10, label=names_lst[i], color=colors[i])

    ax.set_ylim(0.01, 1)

    #ax.set_xscale('log')
    ax.set_yscale('log')

    if ptype == 'grid':
        ax.set_xlim(0, lmax)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    else:
        ax.set_xlabel(r'strength')
        ax.set_ylabel(f'cumulative degree distribution')

# ----------------------------------------------------------------------------------------------------------------------
# Scatter plot:
# ----------------------------------------------------------------------------------------------------------------------

def plot_scatter(ax, data_lst, names_lst, lmax, ptype):
    data1 = data_lst[0].reshape(-1)
    data2 = data_lst[1].reshape(-1)
    ax.scatter(data1, data2, s=10, c="k")

    ax.set_xscale('log')
    ax.set_yscale('log')

    if ptype == 'grid':
        ax.set_xlim(np.min(data_lst[0]), lmax)
        ax.set_ylim(np.min(data_lst[0]), lmax)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(f'{names_lst[0]}')
        ax.set_ylabel(f'{names_lst[1]}')

# ----------------------------------------------------------------------------------------------------------------------
# Other:
# ----------------------------------------------------------------------------------------------------------------------

def plot_line(df, savename, dpi=200):

    plt.rcParams.update({'font.size': 50})

    fig, ax = plt.subplots(figsize=(20, 20))

    sns.lineplot(df, ax=ax, x=df.columns[0], y=df.columns[1], marker='o', markersize=15, linewidth=5, color="black", errorbar="sd")

    ax.set_xlabel(f'{df.columns[0]}')
    ax.set_ylabel(f'{df.columns[1]}')

    ax.set_yscale('log')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_hist(df, savename, dpi=200, n_bins=25):

    plt.rcParams.update({'font.size': 50})

    fig, ax = plt.subplots(figsize=(20, 20))

    sns.histplot(df, ax=ax, x=df.columns[1], hue=df.columns[0], element="step", palette=sns.color_palette(), bins=n_bins)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_ylabel(r'counts')
    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

