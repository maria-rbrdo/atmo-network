"""
========================================================================================================================
BOB Model 2D Data Handling Script
========================================================================================================================
This script reads the individual 2D output files and stores them together in an h5 file.
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
import h5py
import numpy as np
from alive_progress import alive_bar

# Parameters for file and field selection ..............................................................................

fld = 'q'           # Field to visualize
it_start = 1000     # First iteration to plot
it_end = 2000       # Last iteration to plot
res = 'T170'        # Resolution ('T2730', 'T1365', 'T682', 'T341', 'T170', 'T85', 'T42')
cstr = '0'          # Frequency parameter for job identification
tsat = '200'        # Amplitude parameter for job identification

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
    folder = os.path.expanduser("../../dataloc/" + job + "/netdata/")
    file = os.path.expanduser("../../dataloc/" + job + "/netdata/" + f"{fld}_{it_start}_{it_end}")
    tmp = np.loadtxt(f'../../dataloc/grids/GRID.{res}', max_rows=(nlat // 2))
elif host == "remotehost":
    folder = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac/" + job + "/netdata/")
    file = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac/" + job + "/netdata/" + f"{fld}_{it_start}_{it_end}")
    tmp = np.loadtxt(f'../../bob/swbob-vac/grids/GRID.{res}', max_rows=(nlat // 2))

# Create folder to save files ..........................................................................................

if not os.path.exists(folder):
    os.mkdir(folder)

try:
    os.remove(file)
    print(f"Previous file '{file}' deleted successfully.")
except:
    pass

# Latitude and longitude ...............................................................................................

# Create array of longitudes for plotting
xs = np.arange(nlon) / nlon * 360

# Load latitude data from grid file
lattmp = tmp[:, 1]
lats = (np.pi/2 - lattmp) * 180 / np.pi
ys = np.concatenate([lats, -np.flip(lats)])

# Get and store data ...................................................................................................

# Plot contours
qxy = np.empty((nlat//2, nlon, it_end-it_start))
with alive_bar(it_end-it_start, force_tty=True) as bar:
    for it in np.arange(it_start, it_end, 1, dtype=int):
        # Get data
        tstr = f"{it:05}"  # Label of the file at that iteration
        qxy[:, :, it-it_start] = get_dataxy(nlon, tstr, job, fld, host=host, swend=False)[:nlat//2,:]
        # Update bar
        bar()

with h5py.File(file, mode='a') as store:
    store.create_dataset("data", data=qxy)
    store.create_dataset("longitude", data=xs)
    store.create_dataset("latitude", data=ys[:nlat//2])
    store.create_dataset("time", data=np.arange(it_start, it_end, 1))

# Other ................................................................................................................

# Center data at mid-longitude
# qxy = np.roll(qxy, nlon // 2, axis=0)
# xs = xs - xs[nlon // 2]
# Substract height from mid-longitude from all longitudes
#   qxy0 = qxy[0, :]
#   for i in range(nlat):
#       qxy[:, i] -= qxy0[i]
