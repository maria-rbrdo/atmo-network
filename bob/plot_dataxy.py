"""
========================================================================================================================
BOB Model 2D Data Visualization Script
========================================================================================================================
This script reads the 2D output files from the model and plots them.
------------------------------------------------------------------------------------------------------------------------
Usage:
    $ python3 plot_dataxy.py
------------------------------------------------------------------------------------------------------------------------
Notes:
- Available fields: ('h', 'd', 'q', 'z')
------------------------------------------------------------------------------------------------------------------------
"""
from get_dataxy import get_dataxy

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from alive_progress import alive_bar

# Parameters for file and field selection ..............................................................................

fld = 'q'           # Field to visualize
it_start = 1000     # First iteration to plot
it_end = 1500       # Last iteration to plot
res = 'T21'         # Resolution ('T2730', 'T1365', 'T682', 'T341', 'T170', 'T85', 'T42')
cstr = '0'          # Frequency parameter for job identification
tsat = '600'        # Amplitude parameter for job identification
levels = 10          # Levels graph
lmin = -1.2         # Min level
lmax = 2.4          # Max level
ptype = "individual"      # Plot type: grid plot or individual plots

job = 'pv50-nu4-urlx' + '.c' + cstr + 'sat' + tsat + '.' + res    # Job name

host = "localhost"   # localhost or remotehost

# Resolution ...........................................................................................................

if res == 'T2730':
    nlon = 8192
elif res == 'T1365':
    nlon = 4096
elif res == 'T682':
    nlon = 2048
elif res == 'T341':
    nlon = 1024
elif res == 'T170':
    nlon = 512
elif res == 'T85':
    nlon = 256
elif res == 'T42':
    nlon = 128
elif res == 'T21':
    nlon = 64

nlat = nlon // 2

# Host .................................................................................................................

if host == "localhost":
    folder = os.path.expanduser("../../dataloc/" + job + "/imgs/")
    tmp = np.loadtxt(f'../../dataloc/grids/GRID.{res}', max_rows=(nlat // 2))
elif host == "remotehost":
    folder = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac/" + job + "/imgs")
    tmp = np.loadtxt(f'../../bob/swbob-vac/grids/GRID.{res}', max_rows=(nlat // 2))

# Create folder to save files ..........................................................................................

if not os.path.exists(folder):
    os.mkdir(folder)

# Latitude and longitude ...............................................................................................

# Create array of longitudes for plotting
xs = np.arange(nlon) / nlon * 360

# Load latitude data from grid file
lattmp = tmp[:, 1]
lats = (np.pi/2 - lattmp) * 180 / np.pi
ys = np.concatenate([lats, -np.flip(lats)])

# Plotting .............................................................................................................

# Extend qxy array for plotting
xs = np.concatenate((xs, np.atleast_1d(360)))

# Initialise plot
if ptype == "individual":
    plt.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(40, 30))
if ptype == "grid":
    plt.rcParams.update({'font.size': 50})
    fig = plt.figure(figsize=(40, 50))

# Plot contours
with alive_bar(it_end-it_start, force_tty=True) as bar:
    for it in np.arange(it_start, it_end, 1, dtype=int):
        # Set projection
        if ptype == "individual":
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
        if ptype == "grid":
            ax = fig.add_subplot((it_end-it_start)//4, 4, (it - it_start + 1), projection=ccrs.Orthographic(0, 90))
        ax.set_global()

        # Get data
        tstr = f"{it:05}"  # Label of the file at that iteration
        qxy = get_dataxy(nlon, tstr, job, fld, host=host, swend=False)
        qxy = np.concatenate([qxy, np.atleast_2d(qxy[:, 0]).T], axis=1)

        # Plot
        filled_c = ax.contourf(xs, ys, qxy, levels, transform=ccrs.PlateCarree(), cmap=sns.color_palette("Spectral_r", as_cmap=True), vmin=0)
        # ax.contour(xs, ys, qxy, 20, transform=ccrs.PlateCarree(), colors='black', linestyles="--")

        if ptype == "individual":
            fig.colorbar(filled_c, orientation='vertical')
            fig.suptitle(f"{it:04} days")
            fig.savefig(f"{folder}/img{it:05}", dpi=200, bbox_inches='tight')
            fig.clear()
        # Update bar
        bar()

if ptype == "grid":
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(filled_c, cax=cbar_ax, orientation='vertical')
    fig.savefig(f"{folder}/img{it_start}_{it_end}", dpi=200, bbox_inches='tight')
    fig.clear()

# Other ................................................................................................................

# Center data at mid-longitude
# qxy = np.roll(qxy, nlon // 2, axis=0)
# xs = xs - xs[nlon // 2]
# Substract height from mid-longitude from all longitudes
#   qxy0 = qxy[0, :]
#   for i in range(nlat):
#       qxy[:, i] -= qxy0[i]
