"""
Plot correlation outputs.

e.g.:
    $ python3 corr/plot_network.py SWE velocity --tau=0.9 data/euler/SWE_corr/CM_SWE_velocity.h5 --output=data/euler/SWE_corr


Usage:
    plot_network.py <model> <task> [--tau=<tau>] <files> [--output=<dir>] [--degree_distribution=<degree_distribution>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/<model>_correlation_networks/frames]
    --tau=<tau>  Correlation threshold [default: 0.9]
    --degree_distribution=<degree_distribution>  Plot the cumulative degree distribution [default: False]

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from docopt import docopt
from network_properties import calc_centrality, calc_prob_distrib

def main(filename, output, model, task, tau, degree_distribution):

    with h5py.File(filename, mode='r') as f:

        theta = f["theta"][:]  # get positions
        phi = f["phi"][:]
        lon = (np.pi - theta) * 180 / np.pi  # define longitude
        lat = (np.pi / 2 - phi) * 180 / np.pi  # define latitude

        folder_name = f"/{model}_{task}_t{int(tau*100)}_s{int(len(f.keys())-2)}/"
        output_path = os.path.dirname(output+folder_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with alive_bar(len(f.keys()) - 2, force_tty=True) as bar:
            for k in set(f.keys()) - {"theta", "phi"}:

                #%% Load data
                cm = f[k][:]  # get correlation data
                cm = abs(cm)  # get correlation matrix
                cm[cm <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times

                #%% Centrality
                savename = output + folder_name + 'write_{:06}.png'.format(times[2])
                centrality = calc_centrality(cm, lon, lat, times, savename)

                #%% Cumulative degree distribution
                if degree_distribution is True:
                    savename = output + folder_name + "degdistrib_" + 'write_{:06}.png'.format(times[2])
                    _ = calc_prob_distrib(centrality, savename)

                #%% Update bar
                bar()

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(filename=args['<files>'], output=args['--output'], model=args['<model>'], task=args['<task>'],
#         tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"))

main("../data/euler/SWE_corr/CM_SWE_velocity.h5", "../data/euler/SWE_corr", "SWE", "velocity", 0.5, degree_distribution=False)