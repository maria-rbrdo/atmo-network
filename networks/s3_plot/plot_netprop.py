"""
========================================================================================================================
Network Evolution Visualisation Script
========================================================================================================================
This script reads the HDF5 data file created in netevol.py and plots graph measures throughout time.
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

import networks.s2_calculate.netprop as netprop
from networks.s2_calculate.netprop import *
from networks.s3_plot.plot import *

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
        if ptype == "individual":
            figsize = (20, 20)
        elif ptype == "grid":
            figsize = (30, 50)

        if measure == "strength" or measure == "distance":
            fig_in = plt.figure(figsize=figsize)
            fig_out = plt.figure(figsize=figsize)
            fig_diff = plt.figure(figsize=figsize)

            norm_out = mpl.colors.Normalize(vmin=0, vmax=lmax)
            norm_in = mpl.colors.Normalize(vmin=0, vmax=np.abs(lmin))
            norm_diff = mpl.colors.Normalize(vmin=lmin, vmax=lmax)

            sm_in = plt.cm.ScalarMappable(cmap=sns.cm.rocket, norm=norm_in)
            sm_out = plt.cm.ScalarMappable(cmap=sns.cm.rocket, norm=norm_out)
            sm_diff = plt.cm.ScalarMappable(cmap=sns.color_palette("icefire", as_cmap=True), norm=norm_diff)
        else:
            fig = plt.figure(figsize=figsize)
            norm = mpl.colors.Normalize(vmin=0, vmax=lmax)
            sm = plt.cm.ScalarMappable(cmap=sns.cm.rocket, norm=norm)
        if extra_plots:
            fig_distrib = plt.figure(figsize=figsize)
            fig_scatter = plt.figure(figsize=figsize)

        # Set axis .....................................................................................................

        if ptype == "individual":
            if extra_plots:
                ax_distrib = fig_distrib.add_subplot(1, 1, 1)
                ax_scatter = fig_scatter.add_subplot(1, 1, 1)
            if measure == "strength" or measure == "distance":
                ax_in = fig_in.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                ax_out = fig_out.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                ax_diff = fig_diff.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                ax_in.set_global()
                ax_out.set_global()
                ax_diff.set_global()
            else:
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
                ax.set_global()

        # Iterate ......................................................................................................

        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            it = 0
            for k in sorted(keys_data):
            # for k in keys_lags:

                # Set axis .............................................................................................
                if ptype == "grid":
                    if extra_plots:
                        ax_distrib = fig_distrib.add_subplot(len(keys_data) // prow, prow, (it + 1))
                        ax_distrib.set_yticklabels([])
                        ax_distrib.set_xticklabels([])

                        ax_scatter = fig_scatter.add_subplot(len(keys_data) // prow, prow, (it + 1))
                        ax_scatter.set_yticklabels([])
                        ax_scatter.set_xticklabels([])
                    if measure == "strength" or measure == "distance":
                        ax_in = fig_in.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                   projection=ccrs.Orthographic(0, 90))
                        ax_out = fig_out.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                     projection=ccrs.Orthographic(0, 90))
                        ax_diff = fig_diff.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                                       projection=ccrs.Orthographic(0, 90))
                        ax_in.set_global()
                        ax_out.set_global()
                        ax_diff.set_global()
                    else:
                        ax = fig.add_subplot(len(keys_data) // prow, prow, (it + 1),
                                             projection=ccrs.Orthographic(0, 90))
                        ax.set_global()

                # Load data ............................................................................................
                print("Loading data...")
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
                print("Calculating...")
                if measure == "strength" or measure == "distance":
                    glon, glat = np.meshgrid(lon, lat)

                    if measure == "strength":
                        net, net_out = calc_strength(am, len(lat), len(lon), min_dist=0, max_dist=np.inf,
                                                     latcorrected=False, lat=glat.reshape(-1), lon=glon.reshape(-1))
                    else:
                        net, net_out = calc_distance(am, len(lat), len(lon), lat=glat.reshape(-1), lon=glon.reshape(-1))
                    print(np.max(net))
                    # generate plot
                    plot_matrix(ax_out, net_out, lat, lon, min=0, max=lmax, levels=50)
                    if not np.all(np.round(net,10) == np.round(net_out,10)):
                        plot_matrix(ax_in, net, lat, lon, min=0, max=lmax, levels=50)
                        plot_matrix(ax_diff, net_out-net, lat, lon, min=lmin, max=lmax, levels=50)

                    if extra_plots is True:
                        plot_cumsum(ax_distrib, [net, net_out], ["in strength", "out strength"], lmax,
                                    ptype, colors=["tab:blue", "tab:red"])
                        plot_scatter(ax_scatter, [net, net_out], ["in strength", "out strength"], lmax,
                                     ptype)
                else:
                    function = getattr(netprop, 'calc_' + measure)
                    net = function(am, len(lat), len(lon))
                    print(np.min(net), np.mean(net), np.max(net))
                    # generate plot
                    plot_matrix(ax, net, lat, lon, levels=50, min=0, max=lmax, my_cmap=sns.cm.rocket)

                # Save fig .............................................................................................

                if ptype == "individual":
                    if extra_plots:
                        ax_distrib.legend()
                        fig_distrib.savefig(savename_distrib, dpi=200, bbox_inches='tight')
                        ax_distrib.cla()

                        fig_scatter.savefig(savename_scatter, dpi=200, bbox_inches='tight')
                        ax_distrib.cla()

                    if measure == "strength" or measure == "distance":
                        cbar_ax_in = fig_in.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_out = fig_out.add_axes([0.93, 0.15, 0.02, 0.7])
                        cbar_ax_diff = fig_diff.add_axes([0.93, 0.15, 0.02, 0.7])

                        cbar_in = fig_in.colorbar(sm_in, cax=cbar_ax_in, orientation='vertical', extend='both')
                        cbar_in.formatter.set_powerlimits((0, 0))
                        cbar_in.remove()
                        #fig_in.suptitle(f"in strength centrality")
                        fig_in.savefig(savename_in, dpi=200, bbox_inches='tight')
                        ax_in.cla()

                        cbar_out = fig_out.colorbar(sm_out, cax=cbar_ax_out, orientation='vertical', extend='both')
                        cbar_out.formatter.set_powerlimits((0, 0))
                        cbar_out.remove()
                        #fig_out.suptitle(f"out strength centrality")
                        fig_out.savefig(savename_out, dpi=200, bbox_inches='tight')
                        ax_out.cla()

                        cbar_diff = fig_diff.colorbar(sm_diff, cax=cbar_ax_diff, orientation='vertical', extend='both')
                        cbar_diff.formatter.set_powerlimits((0, 0))
                        cbar_diff.remove()
                        #fig_diff.suptitle(f"(in-out) strength centrality")
                        fig_diff.savefig(savename_diff, dpi=200, bbox_inches='tight')
                        ax_diff.cla()

                    else:
                        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])

                        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
                        cbar.remove()
                        #fig.suptitle(f"{measure}")
                        fig.savefig(savename, dpi=200, bbox_inches='tight')
                        ax.cla()

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

            if measure == "strength" or measure == "distance":
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

            else:
                cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
                cbar.formatter.set_powerlimits((0, 0))
                fig.subplots_adjust(wspace=0.1, hspace=0.1)
                fig.savefig(opath + oname + "grid", dpi=200, bbox_inches='tight')

ss = [100, 200, 400, 600, 800, 1000, 1200]
type = "1000_lagged"
if type == "1000_zero": # rho = 0.1
    thresh = {100: 0.455, 200: 0.837, 400: 0.324, 600: 0.373, 800: 0.331, 1000: 0.599, 1200: 0.109}
    times = {100: (1000, 2000), 200: (1000, 2000), 400: (1000, 2000), 600: (1000, 2000), 800: (1000, 2000),
             1000: (1000, 2000), 1200: (1000, 2000)}
elif type == "1000_lagged":  # rho = 0.1
    thresh = {100: 0.505, 200: 0.787, 400: 0.267, 600: 0.293, 800: 0.274, 1000: 0.458, 1200: 0.155}
    times = {100: (1000, 2000), 200: (1000, 2000), 400: (1000, 2000), 600: (1000, 2000), 800: (1000, 2000),
             1000: (1000, 2000), 1200: (1000, 2000)}
elif type == "window_zero":  # rho = 0.05
    thresh = {100: 0.695, 200: 0.751, 400: 0.601, 600: 0.526, 800: 0.542, 1000: 0.597, 1200: 0.508}
    times = {100: (1700, 2000), 200: (1700, 2000), 400: (1200, 1500), 600: (1600, 1900), 800: (1150, 1450),
             1000: (1450, 1750), 1200: (1700, 2000)}
elif type == "window_lagged":
    thresh = {600: 0.712}  # rho = 0.05
    times = {600: (1600, 1900)}
elif type == "vort":
    thresh = {100: 0, 200: 0, 400: 0, 600: 0, 800: 0, 1000: 0, 1200: 0}
    times = {100: (1965, 2000), 200: (1965, 2000), 400: (1325, 1360), 600: (1855, 1890), 800: (1185, 1220),
             1000: (1515, 1550), 1200: (1965, 2000)}
    times = {600: (1600, 1900)}

for s in ss:
    #file = f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/VM_{times[s][0]}_{times[s][1]}.h5"
    #ofile = f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/VM_{times[s][0]}_{times[s][1]}_img"
    file = f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_w1000_s0_l0to7_{times[s][0]}_{times[s][1]}.h5"
    ofile = f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_w1000_s0_l0to7_{times[s][0]}_{times[s][1]}.h5"
    #file = f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_s1_l0to0_{times[s][0]}_{times[s][1]}.h5"
    #ofile = f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_s1_l0to0_{times[s][0]}_{times[s][1]}.h5"
    main(file, ofile, "closeness", thresh[s],
         extra_plots=False, ptype="individual", prow=5, lmin=-0.45, lmax=0.45)

    #main(f"../../../dataloc/netcdf/netdata/VM.h5",
    #     f"../../../dataloc/netcdf/netdata/", "strength", 0,
    #     extra_plots=True, ptype="lgrid", prow=5, lmin=-3e11, lmax=3e11)