"""
========================================================================================================================
Network Evolution Visualization Script
========================================================================================================================
This script plots the evolution of link strength, link distance, and density over the given graphs.
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

from networks.calculate.netprop import *
from networks.plot.plot import *

dpi = 200
ss = [100, 200, 400, 600, 800, 1000, 1200]
s = 600
seg = 40
l = 0
tau = 0

# ------------------------------------------------------------------------------------------------------------------
# Prepare directory:
# ------------------------------------------------------------------------------------------------------------------
fnames = [f"/Volumes/Maria/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/VM_1750_1800.h5"]
opath = f"/Volumes/Maria/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/"
oname = f"{fnames[0].split("/")[-1].split(".")[0] + f"_evol"}/"
folder = os.path.join(opath + oname)
if not os.path.exists(folder):
    os.mkdir(folder)

# Initialise arrays ....................................................................................................
vals = []
time = []

# Iterate ..............................................................................................................
for fname in fnames:
    with h5py.File(fname, mode='r') as f:

        # Load data ....................................................................................................

        lat = f["latitude"][:]
        lon = f["longitude"][:]
        glon, glat = np.meshgrid(lon, lat)
        glon, glat = glon.reshape(-1), glat.reshape(-1)

        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"longitude", "latitude"} - keys_lags

        # Iterate ......................................................................................................
        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            for k in sorted(keys_data):
                # Load data ............................................................................................
                am = f[k][:]  # get data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times
                if len(times) > 3:
                    time.append((times[0]+times[1])/2)
                else:
                    time.append(times[0])

                # Measure and store ....................................................................................

                # Strength
                net, _ = calc_strength(am, len(lat), len(lon), min_dist=0, max_dist=np.inf,
                                             latcorrected=False, lat=glat, lon=glon)
                vals.append(net.reshape(-1))

                # Clustering
                #clust = calc_clustering(am)
                #vals.append(clust)

                #%% Update bar
                bar()

# Plot .........................................................................................................

plt.rcParams.update({'font.size': 30})
fig = plt.figure(figsize=(20, 7), dpi=200)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$t$ (days)')
ax.set_ylabel('global clustering coefficient')
#ax.axvspan(1773,1797,color="gray",alpha=0.3)
#ax.axvspan(1862,1893,color="gray",alpha=0.3)
ax.boxplot(vals, positions=[t for t in time], showfliers=False, widths=1.5, boxprops=dict(linewidth=2),
           whiskerprops=dict(linewidth=2), manage_ticks=False, medianprops=dict(color="black", linewidth=2))
ax.plot([t for t in time], [np.mean(l) for l in vals], "k*")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 1])
fig.tight_layout()
fig.savefig(opath + oname + "evol", dpi=200, bbox_inches='tight')