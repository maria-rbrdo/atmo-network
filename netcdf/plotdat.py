"""
========================================================================================================================
NETCDF Data Visualization Script
========================================================================================================================
This script reads the 2D output files from the model and plots them.
------------------------------------------------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from scipy.io import netcdf_file
from alive_progress import alive_bar

import datetime
from math import ceil

# Parameters for file and field selection ..............................................................................
fpath = ("../../../data/ERA5/6-hourly-data/"
         "ssw.nc")      #File path
host = "localhost"      # localhost or remotehost
lmax = 13.5             # Max level
lmin = 0                # Min level
mzav = False            # Take out mean?
levels = 50             # Levels graph
ptype = "individual"          # Plot type: grid or individual plots
prow = 5                # Number of plots per line if grid


# Output file name .....................................................................................................
fname = "img_"+fpath.split("/")[-1].split(".")[0]

# Output folder ........................................................................................................
if host == "localhost":
    if ptype == "individual":
        folder = os.path.dirname(f"../../../output/ERA5/{fname}/")
    elif ptype == "grid":
        folder = os.path.dirname("../../../output/ERA5/")
elif host == "remotehost":
    folder = os.path.abspath("/home/reboredoprad/bob/dataloc/netcdf/imgs/")

if not os.path.exists(folder):
    os.mkdir(folder)

# Get data .............................................................................................................
f = netcdf_file(fpath, 'r', mmap=False)
lat = f.variables['latitude'][:]
lon = f.variables['longitude'][:]
time = f.variables['time'][:]
pv = (f.variables['pv'][:] * f.variables['pv'].scale_factor + f.variables['pv'].add_offset)/0.0001

# Extend qxy array for plotting
lon = np.concatenate((lon, np.atleast_1d(360)))
pv = np.concatenate([pv, np.atleast_3d(pv[:, :, 0])], axis=2)

# Plotting .............................................................................................................
tit = pv.shape[0]
dt = 1

# Initialise plot

norm = mpl.colors.Normalize(vmin=lmin, vmax=lmax)
sm = plt.cm.ScalarMappable(cmap=sns.color_palette("Spectral_r", as_cmap=True), norm=norm)

# Initialise plot
plt.rcParams.update({'font.size': 100})
if ptype == "individual":
    fig = plt.figure(figsize=(40, 30))
elif ptype == "grid":
    fig = plt.figure(figsize=(40, 10*ceil(tit/dt/prow)))
    norm = mpl.colors.Normalize(vmin=lmin, vmax=lmax)
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette("Spectral_r", as_cmap=True), norm=norm)

# Plot contours
with alive_bar(int(tit//dt), force_tty=True) as bar:
    for i, it in enumerate(np.arange(0, tit, dt)):

        # Set projection
        if ptype == "individual":
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 50))
        elif ptype == "grid":
            ax = fig.add_subplot(int(ceil(tit/dt/prow)), prow, i+1, projection=ccrs.Orthographic(0, 90))
        ax.set_global()
        ax.coastlines(linewidth=2, color="white")

        # Subtract zonal average
        if mzav:
            pv = pv - np.mean(pv, axis=2, keepdims=True)

        # Plot
        filled_c = ax.contourf(lon, lat, pv[it, :, :], levels, transform=ccrs.PlateCarree(),
                               cmap=sns.color_palette("Spectral_r", as_cmap=True), vmin=lmin, vmax=lmax)

        print(np.min(pv[it, :, :]), np.max(pv[it, :, :]))

        # Save img
        if ptype == "individual":
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.75])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend="both")
            cbar.set_label(r'PV (0.0001 K m$^2$ kg$^{-1}$ s$^{-1}$)')
            #cbar.remove()
            #fig.suptitle(f"{it:04} days")
            fig.savefig(f"{folder}/img{it:08.3f}.png", dpi=200, bbox_inches='tight')
            fig.clear()

        # Update bar
        bar()

# Save img
if ptype == "grid":
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.75])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend="both")
    cbar.set_label(r'PV (0.0001 K m$^2$ kg$^{-1}$ s$^{-1}$)')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f"{folder}/{fname}", dpi=200, bbox_inches='tight')
    fig.clear()