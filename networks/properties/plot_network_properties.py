"""
========================================================================================================================
Network Visualization Script
========================================================================================================================
This script builds a graph with the adjacency matrix provided, finds a network measure, and plots it.
------------------------------------------------------------------------------------------------------------------------
Usage:
    indiv_network.py <files> <measure> [--tau=<tau>] [--degree_distribution=<degree_distribution>]

Options:
    --tau=<tau>  Correlation threshold [default: 0.9]
    --degree_distribution=<degree_distribution>  Plot the cumulative degree distribution [default: False]
------------------------------------------------------------------------------------------------------------------------
Notes:

------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
from alive_progress import alive_bar
from docopt import docopt
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import network_properties as net_prop
from network_properties import *
from plotting import *

dpi = 200
def main(fname, opath, measure, tau=0, prob_distrib=False, ptype="individual", prow=5,
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
        prob_distrib: bool, optional    If true plots probability distribution [default: False].
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

        if prob_distrib is True:
            df = pd.DataFrame(columns=["t", "vals"])

        # Initialise plot ..............................................................................................
        plt.rcParams.update({'font.size': 50})
        if ptype == "individual":
            figsize = (20, 20)
        elif ptype == "grid":
            figsize = (30, 50)


        if measure == "centrality":
            fig_in = plt.figure(figsize=figsize)
            fig_out = plt.figure(figsize=figsize)
            fig_diff = plt.figure(figsize=figsize)

            norm_in = mpl.colors.Normalize(vmin=0, vmax=lmax)
            norm_out = mpl.colors.Normalize(vmin=0, vmax=np.abs(lmin))
            norm_diff = mpl.colors.Normalize(vmin=lmin, vmax=lmax)

            sm_in = plt.cm.ScalarMappable(cmap=sns.cm.rocket, norm=norm_in)
            sm_out = plt.cm.ScalarMappable(cmap=sns.cm.mako, norm=norm_out)
            sm_diff = plt.cm.ScalarMappable(cmap=sns.color_palette("icefire", as_cmap=True), norm=norm_diff)
        elif measure == "average_connections":
            fig_in = plt.figure(figsize=figsize)
            fig_out = plt.figure(figsize=figsize)

            norm_in = mpl.colors.Normalize(vmin=0, vmax=lmax)
            norm_out = mpl.colors.Normalize(vmin=lmin, vmax=0)

            sm_in = plt.cm.ScalarMappable(cmap=sns.cm.rocket, norm=norm_in)
            sm_out = plt.cm.ScalarMappable(cmap=sns.cm.mako_r, norm=norm_out)
        else:
            fig = plt.figure(figsize=figsize)
            norm = mpl.colors.Normalize(vmin=0, vmax=lmax)
            sm = plt.cm.ScalarMappable(cmap=sns.color_palette("icefire", as_cmap=True), norm=norm)

        # Iterate ......................................................................................................

        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            it = 0
            for k in sorted(keys_data):
            # for k in keys_lags:

                # Set axis .............................................................................................
                if ptype == "individual":
                    if measure == "centrality":
                        ax_in = fig_in.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                        ax_out = fig_out.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                        ax_diff = fig_diff.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                        ax_in.set_global()
                        ax_out.set_global()
                        ax_diff.set_global()
                    elif measure == "average_connections":
                        ax_in = fig_in.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                        ax_out = fig_out.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                        ax_in.set_global()
                        ax_out.set_global()
                    else:
                        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                        ax.set_global()
                elif ptype == "grid":
                    if measure == "centrality":
                        ax_in = fig_in.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                   projection=ccrs.Orthographic(0, 90))
                        ax_out = fig_out.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                     projection=ccrs.Orthographic(0, 90))
                        ax_diff = fig_diff.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                       projection=ccrs.Orthographic(0, 90))
                        ax_in.set_global()
                        ax_out.set_global()
                        ax_diff.set_global()
                    elif measure == "average_connections":
                        ax_in = fig_in.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                   projection=ccrs.Orthographic(0, 90))
                        ax_out = fig_out.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                     projection=ccrs.Orthographic(0, 90))
                        ax_in.set_global()
                        ax_out.set_global()
                    else:
                        ax = fig.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                             projection=ccrs.Orthographic(0, 90))
                        ax.set_global()

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

                # Measure and plot .....................................................................................

                if measure == "centrality":
                    net, net_out = calc_centrality(am, len(lat), len(lon), min_dist=0, max_dist=np.inf)
                    # generate plot
                    plot_matrix(ax_in, net, lat, lon, min=lmin, max=lmax)
                    plot_matrix(ax_out, -net_out, lat, lon, min=lmin, max=lmax)
                    plot_matrix(ax_diff, net-net_out, lat, lon, min=lmin, max=lmax)
                elif measure == "average_connections":
                    net, net_out = calc_average_connections(am, len(lat), len(lon), min_dist=0, max_dist=np.inf)
                    # generate plot
                    plot_matrix(ax_in, net, lon, lat, min=lmin, max=lmax)
                    plot_matrix(ax_out, -net_out, lon, lat, min=lmin, max=lmax)
                else:  # measures: centrality, clustering, closeness, betweeness, eigenvector
                    function = getattr(net_prop, 'calc_' + measure)
                    net = function(am, len(lat), len(lon))
                    # generate plot
                    plot_matrix(ax, net, lat, lon, min=lmin, max=lmax)

                # Other ................................................................................................

                if ptype == "individual":
                    if measure == "centrality":
                        cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_diff = fig_diff.add_axes([0.93, 0.15, 0.02, 0.7])

                        cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical', extend='both')
                        cbar_in.formatter.set_powerlimits((0, 0))
                        fig_in.suptitle(f"{times[0]:04} days")
                        fig_in.savefig(savename_in, dpi=200, bbox_inches='tight')
                        fig_in.clear()

                        cbar_out = fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical', extend='both')
                        cbar_out.formatter.set_powerlimits((0, 0))
                        fig_out.suptitle(f"{times[0]:04} days")
                        fig_out.savefig(savename_out, dpi=200, bbox_inches='tight')
                        fig_out.clear()

                        cbar_diff = fig_diff.colorbar(sm_diff, cax=cbar_ax_diff, orientation='vertical', extend='both')
                        cbar_diff.formatter.set_powerlimits((0, 0))
                        fig_diff.suptitle(f"{times[0]:04} days")
                        fig_diff.savefig(savename_diff, dpi=200, bbox_inches='tight')
                        fig_diff.clear()

                    elif measure == "average_connections":
                        cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])

                        fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical', extend='both')
                        fig_in.suptitle(f"{times[0]:04} days")
                        fig_in.savefig(savename_in, dpi=200, bbox_inches='tight')
                        fig_in.clear()

                        fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical', extend='both')
                        fig_out.suptitle(f"{times[0]:04} days")
                        fig_out.savefig(savename_out, dpi=200, bbox_inches='tight')
                        fig_out.clear()
                    else:
                        cbar_ax = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])

                        fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
                        fig.suptitle(f"{times[0]:04} days")
                        fig.savefig(savename_diff, dpi=200, bbox_inches='tight')
                        fig.clear()

                # Save data ............................................................................................
                if prob_distrib is True:
                    vals = net.reshape(-1)
                    new_rows = pd.DataFrame({"time": [times[0]] * len(vals), "strength": vals})
                    df = new_rows.copy() if df.empty else pd.concat([df, new_rows], ignore_index=True)

                #%% Update bar
                bar()
                it += 1

        if ptype == "grid":
            if measure == "centrality":
                cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar_ax_diff = fig_diff.add_axes([0.93, 0.15, 0.02, 0.7])

                cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical')
                cbar_in.formatter.set_powerlimits((0, 0))
                fig_in.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_in.savefig(opath + oname + "grid_in", dpi=200, bbox_inches='tight')
                fig_in.clear()

                cbar_out = fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical')
                cbar_out.formatter.set_powerlimits((0, 0))
                fig_out.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_out.savefig(opath + oname + "grid_out", dpi=200, bbox_inches='tight')
                fig_out.clear()

                cbar_diff = fig_diff.colorbar(sm_diff, cax=cbar_ax_diff, orientation='vertical')
                cbar_diff.formatter.set_powerlimits((0, 0))
                fig_diff.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_diff.savefig(opath + oname + "grid_diff", dpi=200, bbox_inches='tight')
                fig_diff.clear()

            elif measure == "average_connections":
                cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])

                cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical')
                cbar_in.formatter.set_powerlimits((0, 0))
                fig_in.subplots_adjust(wspace=0.1, hspace=0.1)
                fig_in.savefig(opath + oname + "grid_in", dpi=200, bbox_inches='tight')
                fig_in.clear()

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
                fig.clear()

        if prob_distrib is True:
            plot_hist_line(df, opath + oname + "prob_distrib.png", n_bins=250)

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])

ss = [600]
seg = 40
l = 7
for s in ss:
    main(f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/CM_q_s{seg}_l0to{l}_1000_2000.h5",
         f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/", "centrality", 0.9,
         prob_distrib=False, ptype="grid", prow=5, lmin=-0.025, lmax=0.025)