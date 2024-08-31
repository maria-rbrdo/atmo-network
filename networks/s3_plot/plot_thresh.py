"""
========================================================================================================================
Threshold Comparison Script
========================================================================================================================
This script reads adjacency matrices stored in HDF5 files and plots the density obtained for different thresholds.
------------------------------------------------------------------------------------------------------------------------
"""

import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from alive_progress import alive_bar
from scipy.interpolate import CubicSpline

from networks.s2_calculate.netprop import *
from networks.s3_plot.plot import *

thresh = np.linspace(0, 1, 50)
ss = [100, 200, 400, 600, 800, 1000, 1200]
mm = ["density", "clustering", "strength", "closeness", "betweenness"]
times = {100: (1700, 2000), 200: (1700, 2000), 400: (1200, 1500), 600: (1600, 1900), 800: (1150, 1450), 1000: (1450, 1750), 1200: (1700, 2000)}
colors = ["tomato", "orange", "mediumseagreen", "darkturquoise", "cornflowerblue", "mediumslateblue", "orchid"]

#s = 600
m = "density"

lmax = 7
window_size = 1000
window_step = 0

pdata = {}

for s in ss:
#for m in mm:

    A0 = s / 600 * 0.15
    print(f"* {A0:04.3f}, {m}:")

    with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_w{window_size}_s{window_step}_l0to{lmax}_1000_2000.h5", mode='r') as f:
    #with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_s1_l0to0_1000_2000.h5", mode='r') as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        nlat, nlon = len(lat), len(lon)

        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"latitude", "longitude"} - keys_lags

        tdata = []
        for k in keys_data:
            print(f"* {k}:")
            #print("Loading...")
            am_org = f[k][:]  # get correlation data
            np.fill_diagonal(am_org, 0)  # take out diagonal
            am_org = np.abs(am_org)  # take absolute value

            kdata = []
            with alive_bar(len(thresh), force_tty=True) as bar:
                for tau in thresh:
                    #print("Thresholding...")
                    am = am_org
                    am[np.abs(am) <= tau] = 0  # impose threshold

                    #print("Calculating...")
                    if m == "strength" or m == "in_strength":
                        net, _ = calc_strength(am, nlat, nlon)
                    elif m == "out_strength":
                        _, net = calc_strength(am, nlat, nlon)
                    elif m == "closeness":
                        net = calc_closeness(am, nlat, nlon)
                    elif m == "betweenness":
                        net = calc_betweenness(am, nlat, nlon)
                    elif m == "clustering":
                        net = calc_clustering(am)
                    elif m == "density":
                        net = calc_density(am)
                    else:
                        raise "Not recognised network measure."

                    #print("Saving...")
                    if isinstance(net, np.ndarray):
                        kdata.append(net.reshape(-1))
                    else:
                        kdata.append(net)

                    bar()
            tdata.append(np.array(kdata).reshape(-1))
        tdata = np.array(tdata).T
        tdata = [[tdata[i, :].tolist() for i in range(tdata.shape[0])]]
        pdata[A0] = tdata

plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(figsize=(11, 12))
plt.hlines(0.1, 0, 1, colors="black", linestyles="--", linewidth=5, label=r"$\rho_\tau$")
for i, k in enumerate(pdata.keys()):
    # threshold
    mean = [np.mean(x) for x in pdata[k][0]]
    #mean = pdata[k]
    cs = CubicSpline(np.flip(mean), np.flip(thresh))
    tau = cs(0.1)
    print(f"{k:04.3f}:{tau:04.3f}")
    # line
    ax.plot(thresh, mean, color=colors[i], linewidth=5, label=f"{float(k):04.3f}H") #
    # threshold
    plt.plot(tau, 0.1, "o", markersize=20, color=colors[i])
box = ax.get_position()
ax.set_position([box.x0+0.01, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
          fancybox=True, shadow=True, ncol=3)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\rho$')
ax.set_yscale('log')
ax.set_xlim(0, 1)

plt.savefig("rhotau.png", dpi=200)
