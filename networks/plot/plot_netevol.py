"""
========================================================================================================================
Network Evolution Visualization Script
========================================================================================================================
This script plots the evolution of link strength, path-length, and density over the given graphs.
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

import networks.calculate.netprop as netprop
from networks.calculate.netprop import *
from networks.plot.plot import *

dpi = 200
def main(fname, opath, measure, tau=0, extra_plots=False, ptype="individual", prow=5,
         lmin=None, lmax=None):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type                  Description
        ------------------------------------------------------------------------------------------
        fpath   : string                Path to the data file.
        opath   : string                Path to the output directory.
        measure : string                Network measure to plot.
        tau : int, optional             Threshold to apply [default: 0].
        extra_plots: bool, optional     If true plots probability distribution and scatter plot [default: False].
        ptype: str, optional            Plot type [default: individual].
        prow: int, optional             Plots per row if grid plot [default: 5].
        ==========================================================================================
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    oname = f"{fname.split("/")[-1].split(".")[0] + f"_{measure}_t{tau}"}/"
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

        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"longitude", "latitude"} - keys_lags

        # Initialise plot ..............................................................................................
        plt.rcParams.update({'font.size': 50})
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)

        # Iterate ......................................................................................................
        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            it = 0
            for k in sorted(keys_data):
                # Load data ............................................................................................
                am = f[k][:]  # get correlation data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times

                # Name output files ....................................................................................
                if ptype == "individual":
                    savename = opath + oname + 'write_{:06}.png'.format(times[-1])
                    savename_in = opath + oname + 'write_{:06}_in.png'.format(times[-1])
                    savename_out = opath + oname + 'write_{:06}_out.png'.format(times[-1])
                    savename_diff = opath + oname + 'write_{:06}_diff.png'.format(times[-1])
                    savename_distrib = opath + oname + 'write_{:06}_distrib.png'.format(times[-1])
                    savename_scatter = opath + oname + 'write_{:06}_scatter.png'.format(times[-1])

                # Measure and plot .....................................................................................

                if measure == "strength":
                    glon, glat = np.meshgrid(lon, lat)
                    net, net_out = calc_strength(am, len(lat), len(lon), min_dist=0, max_dist=1000,
                                                 latcorrected=False, lat=glat.reshape(-1), lon=glon.reshape(-1))
                    # generate plot
                    plot_matrix(ax_out, net_out, lat, lon, min=0, max=lmax, levels=100)
                    if not np.all(np.round(net,10) == np.round(net_out,10)):
                        plot_matrix(ax_in, -net, lat, lon, min=0, max=lmax, levels=100)
                        plot_matrix(ax_diff, net_out-net, lat, lon, min=lmin, max=lmax, levels=100)

                    if extra_plots is True:
                        plot_cumsum(ax_distrib, [net, net_out], ["in strength", "out strength"], lmax,
                                    ptype=ptype)
                        plot_scatter(ax_scatter, [net, net_out], ["in strength", "out strength"], lmax,
                                     ptype=ptype)

                elif measure == "average_connections":
                    net, net_out = calc_average_connections(am, len(lat), len(lon), min_dist=0, max_dist=1000)
                    # generate plot
                    plot_matrix(ax_in, net, lon, lat, min=0, max=lmax)
                    plot_matrix(ax_out, -net_out, lon, lat, min=0, max=lmax)

                else:  # measures: strength, clustering, closeness, betweeness, eigenvector
                    function = getattr(netprop, 'calc_' + measure)
                    net = function(am, len(lat), len(lon))
                    # generate plot
                    plot_matrix(ax, net, lat, lon)

                # Save fig .............................................................................................

                if ptype == "individual":
                    if extra_plots:
                        ax_distrib.legend()
                        fig_distrib.savefig(savename_distrib, dpi=200, bbox_inches='tight')
                        fig_distrib.clear()

                        fig_scatter.savefig(savename_scatter, dpi=200, bbox_inches='tight')
                        fig_distrib.clear()

                    if measure == "strength":
                        cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_diff = fig_diff.add_axes([0.93, 0.15, 0.02, 0.7])

                        cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical', extend='both')
                        cbar_in.formatter.set_powerlimits((0, 0))
                        fig_in.suptitle(f"in strength centrality")
                        fig_in.savefig(savename_in, dpi=200, bbox_inches='tight')
                        fig_in.clear()

                        cbar_out = fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical', extend='both')
                        cbar_out.formatter.set_powerlimits((0, 0))
                        fig_out.suptitle(f"out strength centrality")
                        fig_out.savefig(savename_out, dpi=200, bbox_inches='tight')
                        fig_out.clear()

                        cbar_diff = fig_diff.colorbar(sm_diff, cax=cbar_ax_diff, orientation='vertical', extend='both')
                        cbar_diff.formatter.set_powerlimits((0, 0))
                        fig_diff.suptitle(f"(in-out) strength centrality")
                        fig_diff.savefig(savename_diff, dpi=200, bbox_inches='tight')
                        fig_diff.clear()

                    elif measure == "average_connections":
                        cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])

                        fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical', extend='both')
                        fig_in.suptitle(f"average number of in-links")
                        fig_in.savefig(savename_in, dpi=200, bbox_inches='tight')
                        fig_in.clear()

                        fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical', extend='both')
                        fig_out.suptitle(f"average number of out-links")
                        fig_out.savefig(savename_out, dpi=200, bbox_inches='tight')
                        fig_out.clear()
                    else:
                        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])

                        fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
                        fig.suptitle(f"{measure}")
                        fig.savefig(savename, dpi=200, bbox_inches='tight')
                        fig.clear()

                #%% Update bar
                bar()
                it += 1

        # Save fig .....................................................................................................

        if ptype == "grid":
            if extra_plots is True:
                fig_distrib.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_distrib.savefig(opath + oname + "distrib", dpi=200, bbox_inches='tight')

                fig_scatter.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_scatter.savefig(opath + oname + "scatter", dpi=200, bbox_inches='tight')

            if measure == "strength":
                cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar_ax_diff = fig_diff.add_axes([0.93, 0.15, 0.02, 0.7])

                cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical')
                cbar_in.formatter.set_powerlimits((0, 0))
                fig_in.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_in.savefig(opath + oname + "grid_in", dpi=200, bbox_inches='tight')

                cbar_out = fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical')
                cbar_out.formatter.set_powerlimits((0, 0))
                fig_out.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_out.savefig(opath + oname + "grid_out", dpi=200, bbox_inches='tight')

                cbar_diff = fig_diff.colorbar(sm_diff, cax=cbar_ax_diff, orientation='vertical')
                cbar_diff.formatter.set_powerlimits((0, 0))
                fig_diff.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_diff.savefig(opath + oname + "grid_diff", dpi=200, bbox_inches='tight')

            elif measure == "average_connections":
                cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])

                cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical')
                cbar_in.formatter.set_powerlimits((0, 0))
                fig_in.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_in.savefig(opath + oname + "grid_in", dpi=200, bbox_inches='tight')

                cbar_out = fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical')
                cbar_out.formatter.set_powerlimits((0, 0))
                fig_out.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_out.savefig(opath + oname + "grid_out", dpi=200, bbox_inches='tight')
                fig_out.clear()

            else:
                cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
                cbar.formatter.set_powerlimits((0, 0))
                fig.subplots_adjust(wspace=0.1, hspace=0.1)
                fig.savefig(opath + oname + "grid", dpi=200, bbox_inches='tight')

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

ss = [600]
seg = 40
l = 0
for s in ss:
    main(f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/VM_1763_1803.h5",
         f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/", "strength", 0,
         extra_plots=True, ptype="grid", prow=5, lmin=-2.5e3, lmax=2.5e3)