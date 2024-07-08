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
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from docopt import docopt
import network_properties as net_prop
from network_properties import *
from plotting import *

dpi = 200
def main(measure, tau, filename, output, prob_distrib=False):

    with h5py.File(filename, mode='r') as f:

        theta = f["theta"][:]  # get positions
        phi = f["phi"][:]
        lon = theta  # define longitude in rad
        lat = (np.pi / 2 - phi)  # define latitude in rad
        lon_matrix = lon.reshape(-1, (lon == 0).sum())
        lat_matrix = lat.reshape(-1, (lon == 0).sum())
        lon_unique = np.rad2deg(np.mean(np.flip(lat_matrix.T, 0), axis=1))

        name = filename.split("/")[-1].split(".")[0]+f"_{measure}_t{tau}"
        folder_name = f"/{name}/"
        output_path = os.path.dirname(output+folder_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        df = pd.DataFrame(columns=["t", "vals"])
        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = sorted(set(f.keys()) - {"theta", "phi"} - keys_lags)

        # settings
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figsize=(25, 10))
        ax_in = fig.add_subplot(1, 4, 1)
        ax_out = fig.add_subplot(1, 4, 2)
        ax_diff = fig.add_subplot(1, 4, 3)

        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            for k in keys_data:
            # for k in keys_lags:

                #%% Load data
                am = f[k][:]  # get correlation data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times

                #%% Measure
                if measure == "centrality":
                    net, net_out = calc_centrality(am, lon, lat, min_dist=0, max_dist=np.inf)
                    net_mean = np.mean(np.flip(net.T, 0), axis=1)
                    net_out_mean = np.mean(np.flip(net_out.T, 0), axis=1)
                    net_diff_mean = np.mean(np.flip((net-net_out).T, 0), axis=1)
                    # generate plot
                    ax_in.plot(net_mean, lon_unique, label=f"{int(times[0]) / 1000} hrs")
                    ax_out.plot(net_out_mean, lon_unique, label=f"{int(times[0]) / 1000} hrs")
                    ax_diff.plot(net_diff_mean, lon_unique, label=f"{int(times[0]) / 1000} hrs")
                else:  # measures: clustering, closeness, betweeness, eigenvector
                    print("Not implemented")

                #%% Update bar
                bar()

            # save figure
            ax_in.set_title("avg in centrality")
            ax_out.set_title("avg out centrality")
            ax_diff.set_title("avg in - out centrality")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            savename = output + folder_name + 'averaged_' + measure + '.png'
            fig.savefig(savename, dpi=dpi, bbox_inches='tight')
            plt.close()

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

u = 10
main("centrality", 0.9,
     f"../../data/euler/SWE_corr/n1e5_u{u}_h120_m64/CM_SWE_vorticity_PCC_s5_l0to24.h5",
     f"../../data/euler/SWE_corr/n1e5_u{u}_h120_m64/",
     prob_distrib=False)