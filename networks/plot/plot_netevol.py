"""
========================================================================================================================
Network Evolution Visualization Script
========================================================================================================================
This script plots the evolution of link strength, link distance, and density over the given graphs.
------------------------------------------------------------------------------------------------------------------------
Usage:
    indiv_network.py <files> [--tau=<tau>]

Options:
    --tau=<tau>  Correlation threshold [default: 0]
------------------------------------------------------------------------------------------------------------------------
Notes:

------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py

import numpy as np
import matplotlib as mpl
from docopt import docopt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from alive_progress import alive_bar

from networks.calculate.netprop import *
from networks.plot.plot import *

def main(fname, opath, measure, tau=0):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name        : Type                  Description
        ------------------------------------------------------------------------------------------
        fpath       : string                Path to the data file.
        opath       : string                Path to the output directory.
        measure     : string                Network measure to plot.
        tau         : int, optional         Threshold to apply [default: 0].
        ==========================================================================================
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    oname = f"{fname.split("/")[-1].split(".")[0]}"
    folder = os.path.join(opath + oname +"_evol/")
    ofile = f"{folder}{oname}_evol_t{tau}.h5"
    if not os.path.exists(folder):
        os.mkdir(folder)

    # ------------------------------------------------------------------------------------------------------------------
    # Run:
    # ------------------------------------------------------------------------------------------------------------------

    # Initialise arrays ................................................................................................

    vals = []
    time = []

    with h5py.File(fname, mode='r') as fdata:

        # Load data ....................................................................................................

        lat = fdata["latitude"][:]
        lon = fdata["longitude"][:]
        nlat, nlon = len(lat), len(lon)
        del lat, lon

        keys_lags = {k for k in fdata.keys() if k.endswith("_lags")}
        keys_data = set(fdata.keys()) - {"longitude", "latitude"} - keys_lags

        # Iterate ......................................................................................................
        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            for k in sorted(keys_data)[0:1]:

                # Load data ............................................................................................
                am = fdata[k][:]  # get data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times
                if len(times) > 2:
                    time.append((times[0]+times[1])/2)
                else:
                    time.append(times[0])

                # Measure and store ....................................................................................

                # Strength
                if measure == "strength" or measure == "in strength":
                    net, _ = calc_strength(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "out strength":
                    _, net = calc_strength(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "eigenvector":
                    net = calc_eigenvector(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "closeness":
                    net = calc_closeness(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "betweeness":
                    net = calc_betweeness(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "clustering":
                    net = calc_clustering(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                else:
                    raise "Not recognised network measure."

                # %% Update bar
                bar()

    with h5py.File(ofile, mode='a') as store:
        store.create_dataset(measure, data=vals)
        store.create_dataset(measure+"_t", data=time)

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

type = "corr"
if type == "corr":
    thresh = {100: 0.761, 200: 0.802, 400: 0.688, 600: 0.697, 800: 0.582, 1000: 0.802, 1200: 0.222}
else:
    thresh = {100: 0, 200: 0, 400: 0, 600: 0, 800: 0, 1000: 0, 1200: 0}

ss = [600]
for s in ss:
    main(f"/Volumes/Maria/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/CM_q_w25_s10_l0to0_1600_1900.h5",
         f"/Volumes/Maria/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/", "strength", tau=thresh[s])