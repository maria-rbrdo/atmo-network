"""
Plot sphere outputs.

Usage:
    plot_corr.py <model> <task> [--tau=<tau>] <files> [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/<model>_correlation_networks/frames]
    --tau=<tau>  Correlation threshold [default: 0.9]

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib
import seaborn as snsp
from alive_progress import alive_bar
from docopt import docopt
import pandas as pd
import seaborn as sns

def main(filename, output, model, task, tau):

    with h5py.File(filename, mode='r') as f:

        theta = f["theta"][:]  # get positions
        phi = f["phi"][:]
        lon = (np.pi - theta) * 180 / np.pi  # define longitude
        lat = (np.pi / 2 - phi) * 180 / np.pi  # define latitude

        folder_name = f"/{model}_corr_frames_t{int(tau*100)}_s{int(len(f.keys())-2)}/"
        output_path = os.path.dirname(output+folder_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with alive_bar(len(f.keys()) - 2) as bar:
            for k in set(f.keys()) - {"theta", "phi"}:

                #%% Load data
                cm = f[k][:]  # get corr matrix

                #%% Take the absolute value and impose threshold and
                cm = abs(cm)
                cm[cm <= tau] = 0  # impose threshold

                #%% Calculate centrality
                centrality = np.sum(cm, 0)/cm.shape[0]
                centrality_matrix = centrality.reshape(-1, (theta == 0).sum())

                #%% Plot centrality

                # plot specifics
                dpi = 200
                title_func = lambda start, end: 't = {:.3f} - {:.3f} hrs'.format(start, end)
                savename_func = lambda write: 'write_{:06}.png'.format(write)

                # create figure
                fig, ax = plt.subplots(figsize=(20, 7))
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                plt.rcParams.update({'font.size': 25})

                # plot
                sns.heatmap(np.flip(centrality_matrix.T, 0), cmap="crest", ax=ax,
                            cbar_kws=dict(use_gridspec=False, location="top", aspect=60, extend='both',
                                          label="normalised centrality", pad=0.01))

                # specify ticks plot
                x_ticks = 9
                y_ticks = 5
                ax.set_xticks(np.linspace(0, len(np.unique(lon)), x_ticks))
                ax.set_xticklabels(np.linspace(-180, 180, x_ticks, dtype=int))
                ax.set_yticks(np.linspace(0, len(np.unique(lat)), y_ticks))
                ax.set_yticklabels(np.linspace(90, -90, y_ticks, dtype=int))

                # add time title
                times = [int(s) for s in k.split('_') if s.isdigit()]
                fig.suptitle(title_func(times[0]/100, times[1]/100))

                # save figure
                savename = output + folder_name + savename_func(times[2])
                fig.savefig(savename, dpi=dpi, bbox_inches='tight')
                fig.clear()

                #%% Update bar
                bar()

if __name__ == "__main__":

    args = docopt(__doc__)

    main(filename=args['<files>'], output=args['--output'], model=args['<model>'], task=args['<task>'],
         tau=float(args['--tau']))
