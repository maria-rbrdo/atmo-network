import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_matrix(matrix, measure, lon, lat, times, savename, dpi=200, my_cmap=None, my_center=None):
    # settings
    plt.rcParams.update({'font.size': 25})

    # select colormap depending on whether all values have the same sign or not
    if my_cmap is None:
        if np.any(matrix < 0) & np.any(matrix > 0):
            my_cmap = "icefire"  # sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
            my_center = 0
        else:
            my_cmap = "mako"

    # make figure
    fig, ax = plt.subplots(figsize=(20, 7))
    sns.heatmap(np.flip(matrix.T, 0), ax=ax, cmap=my_cmap, center=my_center,
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