"""
========================================================================================================================
Network Evolution Visualisation Script
========================================================================================================================
This script plots graph measures throughout time.
------------------------------------------------------------------------------------------------------------------------
"""
import h5py
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

type = "vort"
ss = [100, 200, 400, 600, 800, 1000, 1200]
colors = ["tomato", "orange", "mediumseagreen", "darkturquoise", "cornflowerblue", "mediumslateblue", "orchid"]
if type == "1000_zero":  # rho = 0.1
    thresh = {100: 0.455, 200: 0.837, 400: 0.324, 600: 0.373, 800: 0.331, 1000: 0.599, 1200: 0.109}
    times = {100: (1000, 2000), 200: (1000, 2000), 400: (1000, 2000), 600: (1000, 2000), 800: (1000, 2000),
             1000: (1000, 2000), 1200: (1000, 2000)}
elif type == "1000_lagged":
    thresh = {100: 0.505, 200: 0.787, 400: 0.267, 600: 0.293, 800: 0.274, 1000: 0.458, 1200: 0.155}
    times = {100: (1000, 2000), 200: (1000, 2000), 400: (1000, 2000), 600: (1000, 2000), 800: (1000, 2000),
             1000: (1000, 2000), 1200: (1000, 2000)}
elif type == "window_zero":  # rho = 0.05
    thresh = {100: 0.695, 200: 0.751, 400: 0.601, 600: 0.526, 800: 0.542, 1000: 0.597, 1200: 0.508}
    times = {100: (1700, 2000), 200: (1700, 2000), 400: (1200, 1500), 600: (1600, 1900), 800: (1150, 1450),
             1000: (1450, 1750), 1200: (1700, 2000)}
elif type == "window_lagged":
    thresh = {}
    times = {100: (1700, 2000), 200: (1700, 2000), 400: (1200, 1500), 600: (1600, 1900), 800: (1150, 1450),
             1000: (1450, 1750), 1200: (1700, 2000)}
elif type == "vort":
    thresh = {100: 0, 200: 0, 400: 0, 600: 0, 800: 0, 1000: 0, 1200: 0}
    times = {100: (1965, 2000), 200: (1965, 2000), 400: (1325, 1360), 600: (1855, 1890), 800: (1185, 1220),
             1000: (1505, 1540), 1200: (1965, 2000)}
splits = {100: [("st", 1700, 2000)],
          200: [("st", 1700, 2000)],
          400: [("g", 1200, 1252), ("s", 1252, 1270), ("g", 1270, 1334), ("s", 1334, 1362), ("g", 1362, 1446), ("s", 1446, 1480), ("g", 1480, 1500)],
          600: [("g", 1600, 1773), ("s", 1773, 1797), ("g", 1797, 1862), ("s", 1861, 1895), ("g", 1895, 1900)],
          800: [("n", 1150, 1200), ("g", 1200, 1375), ("s", 1375, 1400), ("n", 1400, 1450)],
          1000: [("n", 1450, 1516), ("g", 1516, 1732), ("s", 1732, 1750)],
          1200: [("n", 1700, 2000)]}
m = "in_distance"
ylabel = r"$\tilde{D}_i^\mathrm{in}$ (10$^6$ m)"
ylim = [-0.5, 6] #[-0.03, 8]  # [-0.05, 0.48]
w = 0.5 #0.5  # 5
ms = 10  # 10  # 20
plt.rcParams.update({'font.size': 30})
fig = plt.figure(figsize=(11, 20))
for i, s in enumerate(ss):
# for i, m in enumerate(measures):
    #with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/"
    #               f"CM_q_w25_s10_l0to0_{times[s][0]}_{times[s][1]}_evol/"
    #               f"/CM_q_w25_s10_l0to0_{times[s][0]}_{times[s][1]}_evol_t{thresh[s]}.h5", mode='r') as fdata:
    with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/"
                       f"VM_{times[s][0]}_{times[s][1]}_evol/"
                       f"VM_{times[s][0]}_{times[s][1]}_evol_t{thresh[s]}.h5", mode='r') as fdata:

        t = fdata[m+"_t"][:]
        ax = plt.subplot(7, 1, i+1)

        for splt in splits[s]:
            if splt[0] == "n":
                ax.axvspan(splt[1], splt[2], alpha=0.25, color='gray')
            if splt[0] == "st":
                ax.axvspan(splt[1], splt[2], alpha=0.25, color='darkseagreen')
            if splt[0] == "g":
                ax.axvspan(splt[1], splt[2], alpha=0.5, color='lemonchiffon')
            if splt[0] == "s":
                ax.axvspan(splt[1], splt[2], alpha=0.25, color='tomato')

        if m != "density" and m != "clustering":
            bplot = ax.boxplot(fdata[m][:].T*10**(-6), positions=t, showfliers=False, widths=w, patch_artist=True, manage_ticks=False,
                               boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5), medianprops=dict(linewidth=4, color="black"))
            ax.plot(t, [np.mean(x)*10**(-6) for x in fdata[m][:]], "k*", markersize=ms)
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[i])
        else:
            ax.scatter(t, fdata[m][:], color=colors[i], s=150, edgecolors='black', linewidth=2.5)
        ax.set_xlim([times[s][0], times[s][1]])
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.yaxis.grid(linewidth=2, linestyle=':', color='black')
        if i == len(ss) - 1:
            ax.set_xlabel(f'time (d)')
            ax.set_xticks(np.linspace(times[s][0], times[s][1], 8, dtype=int))
            ax.set_xticklabels(np.linspace(0, 35, 8, dtype=int))
        else:
            ax.set_xticklabels([])

fig.suptitle(r"(c)")
fig.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.075, hspace=0, wspace=0)
plt.savefig("evol.png", dpi=300)