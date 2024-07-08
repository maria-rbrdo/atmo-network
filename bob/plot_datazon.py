"""
========================================================================================================================
BOB Model 1D Data Visualisation Script
========================================================================================================================
This script reads the zonal output files from the model and plots them.
------------------------------------------------------------------------------------------------------------------------
Usage:
    $ python3 plot_datazon.py
------------------------------------------------------------------------------------------------------------------------
Notes:
- Available fields: ('h', 'd', 'q', 'z')
------------------------------------------------------------------------------------------------------------------------
"""

from get_datazon import get_datazon

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alive_progress import alive_bar

# Parameters for file and field selection ..............................................................................

fld = 'u'           # Field to visualize
nt = 1000           # Number of time-steps to consider
res = 'T170'        # Resolution ('T2730', 'T1365', 'T682', 'T341', 'T170', 'T85', 'T42')
cstr = '0000'       # Frequency parameter for job identification
tsat = '600'        # Amplitude parameter for job identification
levels = 100        # Levels graph

job = 'pv50-nu4-urlx' + '.c-' + cstr + 'sat' + tsat + '.' + res    # Job name

host = "localhost"  # localhost or remotehost


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

nlat = nlon // 2

# Host .................................................................................................................

if host == "localhost":
    tmp = np.loadtxt(f'../../dataloc/grids/GRID.{res}', max_rows=(nlat//2))
elif host == "remotehost":
    tmp = np.loadtxt(f'../../bob/swbob-vac/grids/GRID.{res}', max_rows=(nlat//2))

# Latitude and time.....................................................................................................

lattmp = tmp[:, 1]
lats = (np.pi/2 - lattmp) * 180 / np.pi
ys = np.concatenate([lats, -np.flip(lats)])
ys = np.degrees(ys)

# Plotting .............................................................................................................

# Initialise plot
plt.rcParams.update({'font.size': 30})
fig = plt.figure(figsize=(40, 10))

# Getting the data
zon = get_datazon(nlat, nt, job, 'u', swend=True)

# Plotting the contour
plt.subplot(1, 2, 1)
plt.contourf(np.arange(zon.shape[1]), ys, zon, levels=levels)
plt.colorbar()
plt.xlabel('t')
plt.ylabel('f')
plt.title('u(f, t)')

# Save and display the plot
plt.savefig("img")
plt.show()

# Close the plot device (if needed)
# plt.close()