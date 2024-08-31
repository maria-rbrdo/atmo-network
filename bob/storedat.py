"""
========================================================================================================================
BOB Model 2D Data Handling Script
========================================================================================================================
This script reads the individual 2D output files and stores them together in an h5 file.
------------------------------------------------------------------------------------------------------------------------
"""

from getdat import get_data

import os
import h5py
import numpy as np
from alive_progress import alive_bar

# Parameters for file and field selection ..............................................................................

fld = 'u'           # Field to visualize
it_start = 475     # First iteration
it_end = 490       # Last iteration
dt = 0.2              # Timestep
res = 'T170'        # Resolution ('T2730', 'T1365', 'T682', 'T341', 'T170', 'T85', 'T42')
cstr = '0'          # Frequency parameter for job identification
tsat = '1200'       # Amplitude parameter for job identification
new = True          # Old or new file
tag = "_highres"
nh = False           # Only northern hemisphere?

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
    folder = os.path.join("/Volumes/Data/dataloc/" + job + tag + "/netdata/")
    file = os.path.join("/Volumes/Data/dataloc/" + job + tag + "/netdata/" + f"{fld}_{it_start}_{it_end}")
    tmp = np.loadtxt(f'../../dataloc/grids/GRID.{res}', max_rows=(nlat // 2))
elif host == "remotehost":
    folder = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac/" + job + tag + "/netdata/")
    file = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac/" + job + tag + "/netdata/" + f"{fld}_{it_start}_{it_end}")
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

qxy = np.empty((nlat, nlon, int((it_end-it_start)/dt)))
with alive_bar(int((it_end-it_start)/dt), force_tty=True) as bar:
    for idx, it in enumerate(np.arange(it_start, it_end, dt)):
        # Get data
        if new:
            tstr = f"{it:09.3f}"  # Label of the file at that iteration
        else:
            tstr = f"{it:05}"
        qxy[:, :, idx] = get_data(nlon, tstr, str(job+tag), fld, host=host, swend=False)
        # Update bar
        bar()

with h5py.File(file, mode='a') as store:
    if nh:
        store.create_dataset("data", data=qxy[:nlat//2, :, :])
        store.create_dataset("longitude", data=xs)
        store.create_dataset("latitude", data=ys[:nlat//2])
        store.create_dataset("time", data=np.arange(it_start, it_end, dt))
    else:
        store.create_dataset("data", data=qxy[:, :, :])
        store.create_dataset("longitude", data=xs)
        store.create_dataset("latitude", data=ys[:])
        store.create_dataset("time", data=np.arange(it_start, it_end, dt))
