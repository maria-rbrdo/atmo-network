import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matrix(matrix, measure, lon, lat, times, savename, dpi=200):
    # settings
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    # select colormap depending on whether all values have the same sign or not
    if np.any(matrix<0) & np.any(matrix>0):
        my_cmap = "icefire"  # sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
        my_center = 0
    else:
        my_cmap = "mako"
        my_center = None

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

    fig.suptitle('t = {:.3f} - {:.3f} hrs'.format(times[0] / 1000, times[1] / 1000))

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_line(savename, df, dpi=200):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 7))

    sns.lineplot(df, ax=ax, x=df.columns[0], y=df.columns[1], marker='o', markersize=5, color="black", errorbar="sd")

    ax.set_xlabel(f'{df.columns[0]}')
    ax.set_ylabel(f'{df.columns[1]}')

    ax.set_yscale('log')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_hist(savename, df, dpi=200):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(figsize=(20, 7))

    sns.histplot(df, ax=ax, x=df.columns[1], hue=df.columns[0], element="step", stat="percent", common_norm=False,
                 palette=sns.color_palette(), bins=50)

    ax.set_ylabel(r'percentage (%)')
    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()