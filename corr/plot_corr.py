"""
Plot sphere outputs.

Usage:
    plot_corr.py <model> <task> [--tau=<tau>] <files> [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/<model>_correlation_networks/frames]
    --tau=<tau>  Correlation threshold [default: 0.9]

"""

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

    print("Opening h5 file...")
    f = h5py.File(filename, 'r')

    with alive_bar(len(f.keys())) as bar:
        for k in f.keys():

            #%% Load data
            print("Loading h5 file...")
            df = pd.read_hdf(filename, key=k)  # read dataframe

            pos = np.array(df.columns.to_list())  # get positions
            lon = (np.pi - pos[:, 0]) * 180 / np.pi  # define longitude
            lat = (np.pi / 2 - pos[:, 1]) * 180 / np.pi  # define latitude

            cm = df.to_numpy()  # get corr matrix

            #%% Impose threshold
            print("Imposing threshold...")
            cm[abs(cm) <= tau] = 0  # impose threshold

            #%% Calculate centrality
            print("Calculating centrality...")
            centrality = np.sum(cm, 0)/cm.shape[0]
            centrality_matrix = centrality.reshape(-1, (pos[:, 0] == 0).sum())

            #%% Plot centrality
            print("Plotting centrality...")
            fig, ax = plt.subplots()
            sns.heatmap(np.flip(centrality_matrix.T, 0), cmap="crest", ax=ax,
                        cbar_kws=dict(use_gridspec=False, location="top", aspect=60, extend='both',
                                      label="normalised centrality", pad=0.01))

            x_ticks = 9
            y_ticks = 5
            ax.set_xticks(np.linspace(0, len(np.unique(lon)), x_ticks))
            ax.set_xticklabels(np.linspace(-180, 180, x_ticks, dtype=int))
            ax.set_yticks(np.linspace(0, len(np.unique(lat)), y_ticks))
            ax.set_yticklabels(np.linspace(-90, 90, y_ticks, dtype=int))

            plt.show()

            #%% Update bar
            bar()

#if __name__ == "__main__":

    #args = docopt(__doc__)

    #main(filename=args['<files>'], output=args['--output'], model=args['<model>'], task=args['<task>'],
    #     tau=args['--tau'])

main(filename='../data/euler/SWE_corr/CM_SWE_velocity.h5', output='../data/euler/SWE_correlation_networks/frames',
     model='SWE', task='velocity', tau=0.9)
