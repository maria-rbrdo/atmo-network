import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import cartopy.crs as ccrs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_matrix(ax, matrix, lat, lon, my_cmap=None, min=None, max=None, levels=25, H=np.NaN):
    # define topography
    if H is not np.NaN:
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
    z = np.concatenate((np.atleast_2d(z[:, 0]).T, z, np.atleast_2d(z[:, -1]).T), axis=1)

    ax.contourf(x, y, z, levels, transform=ccrs.PlateCarree(), cmap=my_cmap, vmin=min, vmax=max)

    if H is not np.NaN:
        X, Y = np.meshgrid(np.deg2rad(x), np.deg2rad(y))
        HB = hb(X, Y)
        ax.contour(x, y, HB, colors='white', linewidths=3, transform=ccrs.PlateCarree())


def plot_line(df, savename, dpi=200):

    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 7))

    sns.lineplot(df, ax=ax, x=df.columns[0], y=df.columns[1], marker='o', markersize=5, color="black", errorbar="sd")

    ax.set_xlabel(f'{df.columns[0]}')
    ax.set_ylabel(f'{df.columns[1]}')

    ax.set_yscale('log')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_hist(df, savename, dpi=200, n_bins=50):

    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 20))

    sns.histplot(df, ax=ax, x=df.columns[1], hue=df.columns[0], element="step",
                 palette=sns.color_palette(), bins=n_bins)

    ax.set_ylabel(r'counts')
    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_hist_line(df, savename, dpi=200, n_bins=50):

    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 20))

    dff = pd.DataFrame(columns=[df.columns[0], "counts", "bin_centers"])

    for i in np.unique(df[df.columns[0]].loc[df[df.columns[0]] != 0]):
        counts, bin_edges = np.histogram(df[df.columns[1]].loc[df[df.columns[0]] == i], bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        new_rows = pd.DataFrame({df.columns[0]: [i] * len(counts), 'counts': counts/np.sum(counts), 'bin_centers': bin_centers})
        dff = new_rows.copy() if dff.empty else pd.concat([dff, new_rows], ignore_index=True)

    sns.lineplot(dff, ax=ax, x="bin_centers", y="counts", hue=df.columns[0], palette=sns.color_palette(), linewidth=5)
    ax.set_ylabel(r'probability')
    ax.set_xlabel(f'{df.columns[1]}')

    ax.set_yscale("log")
    ax.set_xscale("log")

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_dist(df, savename, dpi=200, n_bins=50):

    # REQUIRES DATA FOR THRESH = 0

    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 20))

    dff = pd.DataFrame(columns=[df.columns[0], "counts", "bin_centers"])

    counts_0, bin_edges = np.histogram(df[df.columns[1]].loc[df[df.columns[0]] == 0], bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in np.unique(df[df.columns[0]].loc[df[df.columns[0]] != 0]):
        counts, _ = np.histogram(df[df.columns[1]].loc[df[df.columns[0]] == i], bins=n_bins)
        counts = counts / counts_0

        new_rows = pd.DataFrame({df.columns[0]: [i]*len(counts), 'counts': counts, 'bin_centers': bin_centers})
        dff = new_rows.copy() if dff.empty else pd.concat([dff, new_rows], ignore_index=True)

    sns.histplot(dff, ax=ax, x="bin_centers", weights="counts", hue=df.columns[0], element="step",
                 palette=sns.color_palette(), bins=n_bins)
    ax.set_ylabel(r'probability')
    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()