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

s = 600
thresh = {600: 0.712}
#thresh = {600: 0}
times = {600: (1600, 1900)}
splits = {600: [("g", 1600, 1773), ("s", 1773, 1797), ("g", 1797, 1862), ("s", 1861, 1895), ("g", 1895, 1900)]}
ylim = [-0.03, 0.45]
w = 4
ms = 10

plt.rcParams.update({'font.size': 40})
fig = plt.figure(figsize=(20, 6))
ax = plt.subplot(1, 1, 1)

with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/"
               f"CM_q_w25_s10_l0to7_{times[s][0]}_{times[s][1]}_evol/"
               f"CM_q_w25_s10_l0to7_{times[s][0]}_{times[s][1]}_evol_t{thresh[s]}.h5", mode='r') as fdata:

#with h5py.File(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/"
#               f"VM_{times[s][0]}_{times[s][1]}_evol/"
#               f"/VM_{times[s][0]}_{times[s][1]}_evol_t{thresh[s]}.h5", mode='r') as fdata:

    try:
        sin = fdata["in_strength"][:]
        tin = fdata["in_strength_t"][:] - 1.75
        sout = fdata["out_strength"][:]
        tout = fdata["out_strength_t"][:] + 1.75
    except:
        sdata = fdata["strength"][:]
        t = fdata["strength_t"][:]

    for splt in splits[s]:
        if splt[0] == "n":
            ax.axvspan(splt[1], splt[2], alpha=0.25, color='gray')
        if splt[0] == "st":
            ax.axvspan(splt[1], splt[2], alpha=0.25, color='darkseagreen')
        if splt[0] == "g":
            ax.axvspan(splt[1], splt[2], alpha=0.5, color='lemonchiffon')
        if splt[0] == "s":
            ax.axvspan(splt[1], splt[2], alpha=0.25, color='tomato')

    try:
        bin = ax.boxplot(sin.T, positions=tin, showfliers=False, widths=w, patch_artist=True, manage_ticks=False,
                         boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5), medianprops=dict(linewidth=4, color="black"))
        ax.plot(tin, [np.mean(x) for x in sin], "k*", markersize=ms)
        for patch in bin['boxes']:
            patch.set_facecolor("cornflowerblue")

        bout = ax.boxplot(sout.T, positions=tout, showfliers=False, widths=w, patch_artist=True, manage_ticks=False,
                         boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5),
                         medianprops=dict(linewidth=4, color="black"))
        ax.plot(tout, [np.mean(x) for x in sout], "k*", markersize=ms)
        for patch in bout['boxes']:
            patch.set_facecolor("tomato")

        ax.legend([bin["boxes"][0], bout["boxes"][0]], [r'$S_i^\mathrm{in}$', r'$S_i^\mathrm{out}$'], loc='center left', bbox_to_anchor=(1, 0.5))
    except:
        b = ax.boxplot(sdata.T[:, :-1], positions=t[:-1], showfliers=False, widths=w, patch_artist=True, manage_ticks=False,
                     boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5),
                     medianprops=dict(linewidth=4, color="black"))
        ax.plot(t[:-1], [np.mean(x) for x in sdata[:-1, :]], "k*", markersize=ms)
        for patch in b['boxes']:
            patch.set_facecolor("darkgray")

        ax.legend([b["boxes"][0]], [r'$S_i$'], loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xlim([times[s][0], times[s][1]-20])
    ax.set_ylim(ylim)
    ax.yaxis.grid(linewidth=2, linestyle=':', color='black')
    ax.set_xlabel(f'time (d)')
    #ax.set_ylabel(r'strength centrality (m s$^{-1}$)')
    ax.set_xticks(np.arange(times[s][0], times[s][1]-25, 50, dtype=int))
    ax.set_xticklabels(np.arange(0, 275, 50, dtype=int))

fig.suptitle(r"(b)")
fig.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.075, hspace=0, wspace=0)
plt.savefig("evol_indiv.png", dpi=300, bbox_inches='tight')