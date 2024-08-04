"""
========================================================================================================================
Threshold Comparison Script
========================================================================================================================
This script builds compares the density / link distance obtained for different thresholds.
------------------------------------------------------------------------------------------------------------------------
"""

import h5py
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from scipy.interpolate import CubicSpline

from networks.calculate.netprop import *
from networks.plot.plot import *

ss = [100, 200, 400, 600, 800, 1000, 1200]
measures = ["av. local clustering coefficient", "av. strength centrality", "av. betweeness centrality"]
m = "density"
thresh = np.linspace(0, 1, 75)
seg = 1
l = 0
dic = {}

for s in ss:

    A0 = s / 600 * 0.15
    print(f"* {A0:04.3f}, {m}:")

    with h5py.File(f"/Volumes/Maria/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/CM_q_s{seg}_l0to{l}_1000_2000.h5", mode='r') as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        glon, glat = np.meshgrid(lon, lat)
        glon, glat = glon.reshape(-1), glat.reshape(-1)

        df = pd.DataFrame(columns=["threshold", m])
        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"latitude", "longitude"} - keys_lags

        with alive_bar(len(thresh), force_tty=True) as bar:
            for tau in thresh:
                for k in keys_data:
                    # load
                    am = f[k][:]  # get correlation data
                    np.fill_diagonal(am, 0)  # take out diagonal
                    am = np.abs(am)  # take absolute value
                    am[np.abs(am) <= tau] = 0  # impose threshold

                    # calculate
                    if m == "distance (km)":
                        values = calc_distances(am, glat, glon)
                    elif m == "density":
                        values = calc_density(am)
                    elif m == "av. local clustering coefficient":
                        values = calc_clustering(am, len(lat), len(lon))
                    elif m == "av. strength centrality":
                        values, _ = calc_strength(am, len(lat), len(lon))
                        values = np.mean(values)
                    elif m == "av. betweeness centrality":
                        values = calc_betweeness(am, len(lat), len(lon))
                        values = np.mean(values)
                    elif m == "av. eigenvector centrality":
                        values = calc_eigenvector(am, len(lat), len(lon))
                        values = np.mean(values)

                    # append
                    try:
                        new_rows = pd.DataFrame({"threshold": [tau]*len(values), m: values})
                    except:
                        new_rows = pd.DataFrame({"threshold": [tau], m: [values]})
                    df = new_rows.copy() if df.empty else pd.concat([df, new_rows], ignore_index=True)
                bar()
    dic[A0] = df

plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(figsize=(11, 12))
cmap = cm.ScalarMappable(cmap='rainbow')
colors = cmap.to_rgba(range(len(dic)))
plt.hlines(0.05, 0, 1, colors="black", linestyles="--", linewidth=5, label=r"$\rho_\tau$")
for i, k in enumerate(dic.keys()):
    df = dic[k]
    sns.lineplot(dic[k], ax=ax, x=df.columns[0], y=df.columns[1], linewidth=5, color=colors[i], errorbar="sd", label=f"{k:04.3f} H")
    cs = CubicSpline(np.flip(df[df.columns[1]]), np.flip(df[df.columns[0]]))
    tau = cs(0.05)
    plt.plot(tau, 0.05, "o", markersize=20, color=colors[i])
    print(f"{k:04.3f}:{tau:04.3f}")
box = ax.get_position()
ax.set_position([box.x0+0.01, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
          fancybox=True, shadow=True, ncol=3)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\rho$')
ax.set_yscale('log')
ax.set_xlim(0, 1)
ax.set_title(r"(c)")

plt.savefig("rhotau.png", dpi=200)