import os
import h5py
import numpy as np
import pandas as pd
import skimage as ski
from docopt import docopt
import matplotlib.pyplot as plt
from alive_progress import alive_bar

#%%
def autocorr(data):
        corrs = []
        with alive_bar(data.shape[0], force_tty=True) as bar:
            for d in data:
                corrs.append(np.correlate(d, d, mode="same")/len(d))
                bar()
        return np.array(corrs)


#%%
plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(figsize=(11, 12))
ax.hlines(0, 0, 500, color="k", linestyles="--", linewidth=2.5)
ax.hlines(1/np.exp(1), 0, 500, color="k", linestyles="--", linewidth=2.5)

ss = [200, 400, 600, 800, 1000, 1200]
colors = ["orange", "mediumseagreen", "darkturquoise", "cornflowerblue", "mediumslateblue", "orchid"]
for i, s in enumerate(ss):
    fpath = f"/Volumes/Data/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/q_1000_2000"
    dsize = 2
    with h5py.File(fpath, mode='r') as f:
        time = f['time'][:]
        nt = len(time)
        data = f["data"][:]
        rdata = ski.measure.block_reduce(data, block_size=(dsize, dsize, 1), func=np.mean)
        rsdata = rdata.reshape(-1, nt)
        ddata = (rsdata - np.mean(rsdata, axis=1).reshape(-1, 1))/np.std(rsdata, axis=1).reshape(-1, 1)
        mcorr = autocorr(ddata)

        time = time[:500]-1000
        mean = np.mean(mcorr, axis=0)[500:]
        Q1 = np.percentile(mcorr, 25, axis=0)[500:]
        Q3 = np.percentile(mcorr, 75, axis=0)[500:]

        ax.plot(time, mean, color=colors[i], linestyle="", marker="o", markersize=2)
        ax.fill_between(time, Q1, Q3, color=colors[i], alpha=0.3, label=s)

ax.set_xlim(0, 500)
ax.legend()
box = ax.get_position()
ax.set_position([box.x0+0.01, box.y0 + 0.01, box.width, box.height])
ax.set_ylabel("autocorrelation")
ax.set_xlabel("time (d)")
plt.show()