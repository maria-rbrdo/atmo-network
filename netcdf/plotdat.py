"""
====================================================================================================
NETCDF Data Visualization Script
====================================================================================================
This script reads the 2D output files from the model and plots them.
----------------------------------------------------------------------------------------------------
"""

import os
from math import ceil

import cartopy.crs as ccrs
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from alive_progress import alive_bar

# Inputs ............................................................................................

print("Select year:")
#YYYY = input()

for YYYY in [2007, 2008, 2009, 2010]:

    print(YYYY)

    # Parameters ........................................................................................
    FIELD = "pv"
    DT = 1
    FPATH = f"../../../output/ERA5/Y{YYYY}-DJFM-daily-NH-850K/data/Y{YYYY}-DJFM-daily-NH-850K.h5"  # File
    DOUT = f"../../../output/ERA5/Y{YYYY}-DJFM-daily-NH-850K/img/{FIELD}/"  # Output path
    LMAX = 1.5  # Max level
    LMIN = 0  # Min level
    MZAV = False  # Take out mean?
    LEVELS = 25  # Levels graph
    PTYPE = "individual"  # Plot type: grid or individual plots
    PROW = 5  # Number of plots per line if grid

    # Execute ..........................................................................................

    # Input file

    if os.path.isfile(FPATH):
        pass
    else:
        raise Exception(f"No data stored at {FPATH} for year {YYYY}")

    # Output folder
    if not os.path.exists(DOUT):
        os.makedirs(DOUT)

    # Get data
    with h5py.File(FPATH, mode="r") as f:
        lat = np.array(f["latitude"])
        lon = np.array(f["longitude"])
        time = np.array(f["time"])
        field = np.array(f[FIELD]) * 1000

    # Close data
    lon = np.concatenate((lon, np.atleast_1d(180)))
    field = np.concatenate((field, np.atleast_3d(field[:,0,:]).transpose(0,2,1)), axis=1)

    # Initialise plot

    tit = field.shape[0]

    plt.rcParams.update({"font.size": 100})
    size = None
    if PTYPE == "individual":
        size = (40, 30)
    elif PTYPE == "grid":
        size = (40, 10 * ceil(tit / DT / PROW))
    fig = plt.figure(figsize=size)
    norm = mpl.colors.Normalize(vmin=LMIN, vmax=LMAX)
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette("Spectral_r", as_cmap=True), norm=norm)

    # Plot
    with alive_bar(int(tit // DT), force_tty=True) as bar:
        for i, it in enumerate(np.arange(0, tit, DT)):

            # Set projection
            axidx = None
            if PTYPE == "individual":
                axidx = (1, 1, 1)
            elif PTYPE == "grid":
                axidx = (int(ceil(tit / DT / PROW)), PROW, i + 1)
            ax = fig.add_subplot(*axidx, projection=ccrs.Orthographic(0, 90))
            ax.set_global()
            ax.coastlines(linewidth=2, color="white")

            # Subtract zonal average
            if MZAV:
                field = field - np.mean(field, axis=2, keepdims=True)

            # Plot
            filled_c = ax.contourf(
                lon,
                lat,
                field[:, :, it],
                LEVELS,
                transform=ccrs.PlateCarree(),
                cmap=sns.color_palette("Spectral_r", as_cmap=True),
                vmin=LMIN,
                vmax=LMAX,
            )

            #print(np.min(field[it, :, :]), np.max(field[it, :, :]))

            # Save img
            if PTYPE == "individual":
                cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.75])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", extend="both")
                cbar.set_label(r"PV (0.0001 K m$^2$ kg$^{-1}$ s$^{-1}$)")
                # cbar.remove()
                # fig.suptitle(f"{it:04} days")
                fig.savefig(f"{DOUT}/img{it:08.3f}.png", dpi=200, bbox_inches="tight")
                fig.clear()

            # Update bar
            bar()

    # Save img
    if PTYPE == "grid":
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.75])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", extend="both")
        cbar.set_label(r"PV (0.0001 K m$^2$ kg$^{-1}$ s$^{-1}$)")
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.savefig(f"{DOUT}/img_{FPATH.split("/")[-1].split(".")[0]}.png", dpi=50, bbox_inches="tight")
        fig.clear()
