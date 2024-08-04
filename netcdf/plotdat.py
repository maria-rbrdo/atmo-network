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

# Parameters for file and field selection ..............................................................................
fpath = "/Volumes/Maria/dataloc/netcdf/netcdf_data.nc"
host = "localhost"   # localhost or remotehost
lmax = 13.5
lmin = 0
mzav = False
levels = 50

# Host .................................................................................................................
if host == "localhost":
    folder = os.path.dirname("/Volumes/Maria/dataloc/netcdf/imgs/")
elif host == "remotehost":
    folder = os.path.abspath("/home/reboredoprad/bob/dataloc/netcdf/imgs/")

# Create folder to save files ..........................................................................................
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
dt = 5

# Initialise plot
plt.rcParams.update({'font.size': 50})
fig = plt.figure(figsize=(40, 13))
norm = mpl.colors.Normalize(vmin=lmin, vmax=lmax)
sm = plt.cm.ScalarMappable(cmap=sns.color_palette("Spectral_r", as_cmap=True), norm=norm)

# Plot contours
with alive_bar(5, force_tty=True) as bar:
    for it in np.array(range(5)):

        # Set projection
        ax = fig.add_subplot(1, 5, it+1, projection=ccrs.Orthographic(0, 90))
        ax.set_global()
        ax.coastlines(linewidth=2, color="white")

        # Subtract zonal average
        if mzav:
            pv = pv - np.mean(pv, axis=2, keepdims=True)

        # Plot
        filled_c = ax.contourf(lon, lat, pv[(it+1)*dt, :, :], levels, transform=ccrs.PlateCarree(),
                               cmap=sns.color_palette("Spectral_r", as_cmap=True), vmin=lmin, vmax=lmax)

        print(np.min(pv[(it+1)*dt, :, :]), np.max(pv[(it+1)*dt, :, :]))

        # Update bar
        bar()

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.75])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend="both")
cbar.set_label(r'PV (0.0001 K m$^2$ kg$^{-1}$ s$^{-1}$)')
fig.subplots_adjust(wspace=0.1, hspace=0.1)
fig.savefig(f"{folder}/grid", dpi=200, bbox_inches='tight')
fig.clear()

# Other ................................................................................................................

# Center data at mid-longitude
# qxy = np.roll(qxy, nlon // 2, axis=0)
# xs = xs - xs[nlon // 2]
# Substract height from mid-longitude from all longitudes
#   qxy0 = qxy[0, :]
#   for i in range(nlat):
#       qxy[:, i] -= qxy0[i]
