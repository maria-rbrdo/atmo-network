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
from network_properties import *

def main(measure, tau, filename, output, prob_distrib=False):

    with h5py.File(filename, mode='r') as f:

        theta = f["theta"][:]  # get positions
        phi = f["phi"][:]
        lon = theta  # define longitude in rad
        lat = (np.pi / 2 - phi)  # define latitude in rad

        name = filename.split("/")[-1].split(".")[0]+f"_{measure}_t{tau}"
        folder_name = f"/{name}/"
        output_path = os.path.dirname(output+folder_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        df = pd.DataFrame(columns=["t", "vals"])
        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"theta", "phi"} - keys_lags

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
                savename = output + folder_name + 'write_{:06}.png'.format(times[-1])
                if measure == "centrality":
                    net, _ = calc_centrality(am, lon, lat, times, savename, min_dist=0, max_dist=np.inf)
                else:  # measures: centrality, clustering, closeness, betweeness, eigenvector
                    function = getattr(net_prop, 'calc_' + measure)
                    net = function(am, lon, lat, times, savename)

                #%% Plot

                #%% Save data
                if prob_distrib is True:
                    vals = net.reshape(-1)
                    new_rows = pd.DataFrame({"time": [times[0]] * len(vals), "strength": vals})
                    df = new_rows.copy() if df.empty else pd.concat([df, new_rows], ignore_index=True)

                #%% Update bar
                bar()

            if prob_distrib is True:
                plot_hist_line(df, output + folder_name + "prob_distrib.png", n_bins=250)

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

u = 80
main("centrality", 0.9,
     f"../../data/euler/SWE_vort/n1e5_u{u}_h120_m64/CM_SWE.h5",
     f"../../data/euler/SWE_vort/n1e5_u{u}_h120_m64/",
     prob_distrib=False)