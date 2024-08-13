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



#fname = "/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat600.T170/CM_q_w25_s10_l0to0_1600_1900_evol/CM_q_w25_s10_l0to0_1600_1900_evol_t0.373.h5"
#fname = "/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat600.T170/VM_1600_1900_evol/VM_1600_1900_evol_t0.h5"
#oname = ".".join(fname.split(".")[:-1])+".png"
#measures = ["density", "clustering", "strength", "out_strength", "closeness", "betweenness"]
#labels = [r"$\rho$", r"$GC$", r"$\langle S_i^\mathrm{in} \rangle$", r"$\langle S_i^\mathrm{out} \rangle$", r"$C_i$", r"$BC_i$ ($10^4$)"]
ss = [100, 200, 400, 600, 800, 1000, 1200]
colors = ["tomato", "orange", "mediumseagreen", "darkturquoise", "cornflowerblue", "mediumslateblue", "orchid"]
times = {100: (1700, 2000), 200: (1700, 2000), 400: (1200, 1500), 600: (1600, 1900),
         800: (1150, 1450), 1000: (1450, 1750), 1200: (1700, 2000)}
thresh = {100: 0.455, 200: 0.837, 400: 0.324, 600: 0.373, 800: 0.331, 1000: 0.599, 1200: 0.109}
splits = {100: [("st", 1700, 2000)],
          200: [("st", 1700, 2000)],
          400: [("g", 1200, 1252), ("s", 1252, 1270), ("g", 1270, 1334), ("s", 1334, 1362), ("g", 1362, 1446), ("s", 1446, 1480), ("g", 1480, 1500)],
          600: [("g", 1600, 1773), ("s", 1773, 1797), ("g", 1797, 1862), ("s", 1862, 1895), ("g", 1895, 1900)],
          800: [("n", 1150, 1200), ("g", 1200, 1375), ("s", 1375, 1400), ("n", 1400, 1450)],
          1000: [("n", 1450, 1530), ("g", 1530, 1732), ("s", 1732, 1750)],
          1200: [("n", 1700, 2000)]}
m = "density"
w = 5.25  # 2.25  # 5
ms = 10  # 10  # 20
plt.rcParams.update({'font.size': 35})
fig = plt.figure(figsize=(11, 20))
for i, s in enumerate(ss):
# for i, m in enumerate(measures):
    with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/"
                   f"CM_q_w25_s10_l0to0_{times[s][0]}_{times[s][1]}_evol/"
                   f"/CM_q_w25_s10_l0to0_{times[s][0]}_{times[s][1]}_evol_t{thresh[s]}.h5", mode='r') as fdata:
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

        if m == "strength":
            bplot = ax.boxplot(fdata[m][:].T, positions=t, showfliers=False, widths=w, patch_artist=True, manage_ticks=False,
                               boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5), medianprops=dict(linewidth=4, color="black"))
            ax.plot(t, [np.mean(x) for x in fdata[m][:]], "k*", markersize=ms)
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[i])
        if m == "density":
            ax.plot(t, fdata[m][:], "k*", markersize=ms)
        ax.set_xlim([times[s][0], times[s][1]])
        ax.set_ylim([-0.2, 0.75])
        ax.set_ylabel(r"$\langle S_i \rangle$")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if i == len(ss) - 1:
            ax.set_xlabel(f'time (d)')
            ax.set_xticklabels(np.linspace(0, 300, 7, dtype=int))
        else:
            ax.set_xticklabels([])

fig.suptitle(r"(c)")
fig.subplots_adjust(left=0.16, right=0.95, top=0.94, bottom=0.075, hspace=0, wspace=0)
plt.savefig("evol.png", dpi=300)