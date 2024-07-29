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

dpi = 200
def main(fname, opath, tau=0, pdensity=False):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name        : Type                  Description
        ------------------------------------------------------------------------------------------
        fpath       : string                Path to the data file.
        opath       : string                Path to the output directory.
        tau         : int, optional         Threshold to apply [default: 0].
        pdensity    : bool, optional        Whether to plot the density evolution [default: False].
        ==========================================================================================
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    oname = f"{fname.split("/")[-1].split(".")[0] + f"_evol_t{tau}"}/"
    folder = os.path.dirname(opath + oname)
    if not os.path.exists(folder):
        os.mkdir(folder)

    # ------------------------------------------------------------------------------------------------------------------
    # Run:
    # ------------------------------------------------------------------------------------------------------------------

    with h5py.File(fname, mode='r') as f:

        # Load data ....................................................................................................

        lat = f["latitude"][:]
        lon = f["longitude"][:]
        glon, glat = np.meshgrid(lon, lat)
        glon, glat = glon.reshape(-1), glat.reshape(-1)

        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"longitude", "latitude"} - keys_lags

        # Initialise arrays ............................................................................................
        lstr = []
        lstr_std = []
        llen = []
        llen_std = []
        if pdensity:
            density = []
            density_std = []
        time = []

        # Iterate ......................................................................................................
        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            for k in sorted(keys_data):
                # Load data ............................................................................................
                am = f[k][:]  # get correlation data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times
                if len(times) > 3:
                    time.append((times[0]+times[1])/2)
                else:
                    time.append(times[0])

                # Measure and store ....................................................................................

                # Strength
                net, net_out = calc_strength(am, len(lat), len(lon), min_dist=0, max_dist=np.inf,
                                             latcorrected=False, lat=glat, lon=glon)
                all_str = np.concatenate((net, net_out))
                lstr.append(np.mean(all_str, axis=(0, 1)))
                lstr_std.append(np.std(all_str, axis=(0, 1)))

                # Distance
                net, net_out = calc_avlen(am, len(lat), len(lon), glat, glon)
                all_len = np.concatenate((net, net_out))
                llen.append(np.mean(all_len, axis=(0, 1)))
                llen_std.append(np.std(all_len, axis=(0, 1)))

                # Density
                if pdensity:
                    all_density = np.calc_density(am)
                    density.append(np.mean(all_density))
                    density_std.append(np.std(all_density))

                #%% Update bar
                bar()

        # Plot .........................................................................................................

        plt.rcParams.update({'font.size': 50})
        fig = plt.figure(figsize=(40, 20), dpi=200)
        ax_str = fig.add_subplot(1, 1, 1)
        ax_dst = ax_str.twinx()

        lstr = np.array(lstr)
        lstr_std = np.array(lstr_std)
        llen = np.array(llen)
        llen_std = np.array(llen_std)

        ax_str.plot(time, lstr, "-", color='tab:red', linewidth=7)
        ax_str.fill_between(time, lstr+lstr_std, lstr-lstr_std, color='tab:red', alpha=0.2)
        ax_str.tick_params(axis='y', labelcolor='tab:red')
        ax_dst.plot(time, llen, "-", color='tab:blue', linewidth=7)
        ax_dst.fill_between(time, llen + llen_std, llen-llen_std, color='tab:blue', alpha=0.2)
        ax_dst.tick_params(axis='y', labelcolor='tab:blue')

        ax_str.set_xlabel('time (days)')
        ax_str.set_ylabel('link strength', color='tab:red')
        ax_dst.set_ylabel('link length', color='tab:blue')

        #if pdensity:
        #    density = np.array(density)
        #    density_std = np.array(density_std)

        #    ax.plot(times, density, "k-", label="Density")
        #    ax.fill_between(times, density + density_std, density-density_std, c="k", alpha=0.2)

        fig.tight_layout()
        fig.savefig(opath + oname + "evol", dpi=200, bbox_inches='tight')

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

ss = [100, 200, 400, 600, 800, 1000, 1200]
ss = [600]
seg = 40
l = 0
for s in ss:
    #main(f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/VM_1763_1803.h5",
    #     f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/", 0)
    main(f"../../../dataloc/netcdf/netdata/data.h5",
         f"../../../dataloc/netcdf/netdata/", 0)