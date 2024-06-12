"""
Plot correlation outputs.

e.g.:
    $ python3 networks/indiv_network.py SWE velocity PCC centrality --lag=0 --tau=0.9 data/euler/SWE_corr/CM_SWE_velocity_PCC_s1_l1.h5 --output=data/euler/SWE_corr


Usage:
    indiv_network.py <model> <task> <method> <measure> [--lag=<lag>] [--tau=<tau>] <files> [--output=<dir>] [--degree_distribution=<degree_distribution>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/<model>_correlation_networks/frames]
    --tau=<tau>  Correlation threshold [default: 0.9]
    --degree_distribution=<degree_distribution>  Plot the cumulative degree distribution [default: False]

"""

import os
import h5py
import numpy as np
from alive_progress import alive_bar
from docopt import docopt
import network_properties as net_prop
from network_properties import calc_prob_distrib

def main(model, task, method, measure, lag, tau, filename, output, degree_distribution):

    with h5py.File(filename, mode='r') as f:

        theta = f["theta"][:]  # get positions
        phi = f["phi"][:]
        lon = (np.pi - theta) * 180 / np.pi  # define longitude
        lat = (np.pi / 2 - phi) * 180 / np.pi  # define latitude

        folder_name = f"/{model}_{task}_{method}_{measure}_s{int(len(f.keys())-2)}_l{lag}_t{int(tau*100)}/"
        output_path = os.path.dirname(output+folder_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with alive_bar(len(f.keys()) - 2, force_tty=True) as bar:
            for k in set(f.keys()) - {"theta", "phi"}:

                #%% Load data
                cm = f[k][:]  # get correlation data
                np.fill_diagonal(cm, 0)  # take out diagonal
                cm = np.abs(cm) # take absolute value
                cm[np.abs(cm) <= tau] = 0  # impose threshold
                #cm = np.where(cm > 0, 1, np.where(cm < 0, -1, 0))  # unweighted matrix

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times

                #%% Centrality
                savename = output + folder_name + 'write_{:06}.png'.format(times[2])
                try:  # measures: centrality, clustering, closeness, betweeness, eigenvector
                    function = getattr(net_prop, 'calc_'+measure)
                    net = function(cm, lon, lat, times, savename, dpi=200)
                except:
                    raise ValueError(f"Unknown measure: {measure}.")

                #%% Cumulative degree distribution
                if degree_distribution is True:
                    savename = output + folder_name + "degdistrib_" + 'write_{:06}.png'.format(times[2])
                    _ = calc_prob_distrib(net, measure, savename)

                #%% Update bar
                bar()

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

main("SWE", "vorticity", "PCC", "centrality", 23,
     0.9, "../../data/euler/SWE_corr/CM_SWE_vorticity_PCC_s5_l23.h5", "../../data/euler/SWE_corr",
     degree_distribution=False)
