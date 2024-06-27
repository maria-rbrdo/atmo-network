import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import cartopy.crs as ccrs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_matrix(matrix, measure, lon, lat, times, savename, dpi=200, my_cmap=None, my_center=None):
    # settings
    plt.rcParams.update({'font.size': 25})

    # select colormap depending on whether all values have the same sign or not
    if my_cmap is None:
        if np.all(matrix >= 0):
            my_cmap = sns.cm.rocket
        elif np.all(matrix <= 0):
            my_cmap = sns.cm.mako_r
        else:
            my_cmap = sns.cm.icefire
    # make figure
    x = np.rad2deg(np.flip(lon.T, 0))
    y = np.rad2deg(np.flip(lat.T, 0))
    z = np.flip(matrix.T, 0)

    x = np.hstack((x, 360*np.ones((len(x[:, 0]), 1))))
    y = np.hstack((y, y[:, 0].reshape(-1, 1)))
    z = np.hstack((z, z[:, 0].reshape(-1, 1)))

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIV(central_longitude=180))
    ax.set_global()
    filled_c = ax.contourf(x, y, z, transform=ccrs.PlateCarree(), cmap=my_cmap)
    ax.contour(x, y, z, levels=filled_c.levels, colors='black', transform=ccrs.PlateCarree())
    fig.colorbar(filled_c, use_gridspec=False, location="bottom", aspect=50, label=f"{measure}", orientation='horizontal', pad=0.05)

    if len(times) == 3:
        fig.suptitle('t = {:.3f} - {:.3f} hrs'.format(times[0] / 1000, times[1] / 1000))
    else:
        fig.suptitle('t = {:.3f} hrs'.format(times[0] / 1000, times[1] / 1000))

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

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

    fig, ax = plt.subplots(figsize=(20, 7))

    sns.histplot(df, ax=ax, x=df.columns[1], hue=df.columns[0], element="step",
                 palette=sns.color_palette(), bins=n_bins)

    ax.set_ylabel(r'counts')
    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_hist_line(df, savename, dpi=200, n_bins=50):

    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 7))

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

    fig, ax = plt.subplots(figsize=(20, 7))

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