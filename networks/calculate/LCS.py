"""
========================================================================================================================
LCS Detection Script
========================================================================================================================
This script performs three splits from the Lagrangian adjacency matrices and plots them.
------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
import scipy as sp

import matplotlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from sklearn.cluster import AgglomerativeClustering

def find_split(A, n):

    D = np.diag(np.sum(A, axis=1))
    #Dinv = np.diag(1/np.sum(A, axis=1))
    L = D - A
    #DinvL = Dinv @ L

    # eigenvalues
    #l1, x1 = sp.linalg.eigs(DinvL, subset_by_index=[A.shape[0] - n, A.shape[0] - 1])
    l, X = sp.linalg.eigh(L, D, subset_by_index=[A.shape[0] - n, A.shape[0] - 1])
    X = np.flip(X, axis=1)  # decreasing order

    # split
    S = np.zeros_like(X)
    for i, x in enumerate(X.T):
        clustering = AgglomerativeClustering(linkage="complete").fit(x.reshape(-1, 1))
        S[:, i] = clustering.labels_

    # from binary to int
    s = np.array([''.join(S[i, :].astype('int').astype('str')) for i in range(A.shape[0])])

    return s

def create_subplot(fig, position, title, projection=ccrs.Orthographic(0, 90), sph=True):
    ax = fig.add_subplot(position[0], position[1], position[2], projection=projection)
    if sph:
        ax.set_global()
        # ax.gridlines(color="k", linestyle=':', linewidth=2.5)
    ax.set_title(title)
    return ax

sph = True
fpath = "/Volumes/Data/dataloc/pv50-nu4-urlx.c0sat1200.T170_highres/netdata/TM_400_425.h5"
#fpath = "/Volumes/Data/dataloc/quadgyre/netdata/TM.h5"
with h5py.File(fpath, mode='r') as f:

    print("Getting coordinates...")
    coord = f["coord"][:]

    print("Clustering...")
    s = find_split(f["adj"][:], 3)

    print("Plotting...")
    # Initialize the figure
    plt.rcParams.update({'font.size': 20})
    if sph:
        fig = plt.figure(figsize=(20, 10))
    else:
        fig = plt.figure(figsize=(15, 10))

    fig_all = plt.figure(figsize=(20, 20))
    fig_000 = plt.figure(figsize=(20, 20))
    fig_001 = plt.figure(figsize=(20, 20))
    fig_010 = plt.figure(figsize=(20, 20))
    fig_011 = plt.figure(figsize=(20, 20))
    fig_100 = plt.figure(figsize=(20, 20))
    fig_101 = plt.figure(figsize=(20, 20))
    fig_110 = plt.figure(figsize=(20, 20))
    fig_111 = plt.figure(figsize=(20, 20))

    if sph:
        prjct = ccrs.Orthographic(0, 90)
    else:
        prjct = None

    # Create the subplots
    ax_all = create_subplot(fig, (1, 2, 1), "All trajectories", projection=prjct, sph=sph)

    ax_0 = create_subplot(fig, (2, 4, 3), "Branch 0", projection=prjct, sph=sph)
    ax_1 = create_subplot(fig, (2, 4, 7), "Branch 1", projection=prjct, sph=sph)

    ax_00 = create_subplot(fig, (4, 8, 8-1), "Branch 00", projection=prjct, sph=sph)
    ax_01 = create_subplot(fig, (4, 8, 8*2-1), "Branch 01", projection=prjct, sph=sph)
    ax_10 = create_subplot(fig, (4, 8, 8*3-1), "Branch 10", projection=prjct, sph=sph)
    ax_11 = create_subplot(fig, (4, 8, 8*4-1), "Branch 11", projection=prjct, sph=sph)

    ax_000 = create_subplot(fig, (8, 16, 16-1), "Branch 000", projection=prjct, sph=sph)
    ax_001 = create_subplot(fig, (8, 16, 16*2-1), "Branch 001", projection=prjct, sph=sph)
    ax_010 = create_subplot(fig, (8, 16, 16*3-1), "Branch 010", projection=prjct, sph=sph)
    ax_011 = create_subplot(fig, (8, 16, 16*4-1), "Branch 011", projection=prjct, sph=sph)
    ax_100 = create_subplot(fig, (8, 16, 16*5-1), "Branch 100", projection=prjct, sph=sph)
    ax_101 = create_subplot(fig, (8, 16, 16*6-1), "Branch 101", projection=prjct, sph=sph)
    ax_110 = create_subplot(fig, (8, 16, 16*7-1), "Branch 110", projection=prjct, sph=sph)
    ax_111 = create_subplot(fig, (8, 16, 16*8-1), "Branch 111", projection=prjct, sph=sph)

    ax_fig_all = create_subplot(fig_all, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_000 = create_subplot(fig_000, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_001 = create_subplot(fig_001, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_010 = create_subplot(fig_010, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_011 = create_subplot(fig_011, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_100 = create_subplot(fig_100, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_101 = create_subplot(fig_101, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_110 = create_subplot(fig_110, (1, 1, 1), None, projection=prjct, sph=sph)
    ax_fig_111 = create_subplot(fig_111, (1, 1, 1), None, projection=prjct, sph=sph)

    color_dict = {"000": "brown", "001": "red", "010": "darkorange", "011": "yellowgreen",
                  "100": "lightseagreen", "101": "royalblue", "110": "blueviolet", "111": "navy"}

    with alive_bar(coord.shape[0], force_tty=True) as bar:
        for n in range(coord.shape[0]):
            branch = s[n]
            color = color_dict[branch]
            ax_list = [ax_all, ax_fig_all]
            if branch[0] == "0":
                ax_list.append(ax_0)
                if branch[1] == "0":
                    ax_list.append(ax_00)
                    if branch[2] == "0":
                        ax_list.append(ax_000)
                        ax_list.append(ax_fig_000)
                    else:
                        ax_list.append(ax_001)
                        ax_list.append(ax_fig_001)
                else:
                    ax_list.append(ax_01)
                    if branch[2] == "0":
                        ax_list.append(ax_010)
                        ax_list.append(ax_fig_010)
                    else:
                        ax_list.append(ax_011)
                        ax_list.append(ax_fig_011)
            else:
                ax_list.append(ax_1)
                if branch[1] == "0":
                    ax_list.append(ax_10)
                    if branch[2] == "0":
                        ax_list.append(ax_100)
                        ax_list.append(ax_fig_100)
                    else:
                        ax_list.append(ax_101)
                        ax_list.append(ax_fig_101)
                else:
                    ax_list.append(ax_11)
                    if branch[2] == "0":
                        ax_list.append(ax_110)
                        ax_list.append(ax_fig_110)
                    else:
                        ax_list.append(ax_111)
                        ax_list.append(ax_fig_111)

            for ax in ax_list:
                if sph:
                    ax.scatter(coord[n, 1, 0], coord[n, 0, 0], c=color, marker="o", s=200, transform=ccrs.Geodetic())
                    ax.plot(coord[n, 1, :], coord[n, 0, :], "-", color=color, alpha=0.25, transform=ccrs.Geodetic(), linewidth=10)
                else:
                    ax.scatter(coord[n, 1, -1], coord[n, 0, 0], c=color, marker="*", s=1)
                    ax.scatter(coord[n, 1, 0], coord[n, 0, 0], c=color, marker="o", s=1)
                    ax.plot(coord[n, 1, :], coord[n, 0, :], "-", color=color, alpha=0.10)
            bar()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    b0 = ax_0.get_position()
    b1 = ax_1.get_position()
    ax_0.set_position([b0.bounds[0] - 0.015, b0.bounds[1], b0.bounds[2], b0.bounds[3]])
    ax_1.set_position([b1.bounds[0] - 0.015, b1.bounds[1], b1.bounds[2], b1.bounds[3]])

    b000 = ax_000.get_position()
    b001 = ax_001.get_position()
    b010 = ax_010.get_position()
    b011 = ax_011.get_position()
    b100 = ax_100.get_position()
    b101 = ax_101.get_position()
    b110 = ax_110.get_position()
    b111 = ax_111.get_position()

    ax_000.set_position([b000.bounds[0] + 0.05, b000.bounds[1], b000.bounds[2], b000.bounds[3] - 0.015])
    ax_001.set_position([b001.bounds[0] + 0.05, b001.bounds[1], b001.bounds[2], b001.bounds[3] - 0.015])
    ax_010.set_position([b010.bounds[0] + 0.05, b010.bounds[1], b010.bounds[2], b010.bounds[3] - 0.015])
    ax_011.set_position([b011.bounds[0] + 0.05, b011.bounds[1], b011.bounds[2], b011.bounds[3] - 0.015])
    ax_100.set_position([b100.bounds[0] + 0.05, b100.bounds[1], b100.bounds[2], b100.bounds[3] - 0.015])
    ax_101.set_position([b101.bounds[0] + 0.05, b101.bounds[1], b101.bounds[2], b101.bounds[3] - 0.015])
    ax_110.set_position([b110.bounds[0] + 0.05, b110.bounds[1], b110.bounds[2], b110.bounds[3] - 0.015])
    ax_111.set_position([b111.bounds[0] + 0.05, b111.bounds[1], b111.bounds[2], b111.bounds[3]])

    path = os.path.dirname(fpath)
    try:
        os.makedirs(path+"/figures/")
    except:
        pass
    fig.savefig(path+"/figures/coloring_all_all.png", dpi=300)
    fig_all.savefig(path + "/figures/coloring_all.png", dpi=300, bbox_inches='tight')
    fig_000.savefig(path + "/figures/coloring_000.png", dpi=300, bbox_inches='tight')
    fig_001.savefig(path + "/figures/coloring_001.png", dpi=300, bbox_inches='tight')
    fig_010.savefig(path + "/figures/coloring_010.png", dpi=300, bbox_inches='tight')
    fig_011.savefig(path + "/figures/coloring_011.png", dpi=300, bbox_inches='tight')
    fig_100.savefig(path + "/figures/coloring_100.png", dpi=300, bbox_inches='tight')
    fig_101.savefig(path + "/figures/coloring_101.png", dpi=300, bbox_inches='tight')
    fig_110.savefig(path + "/figures/coloring_110.png", dpi=300, bbox_inches='tight')
    fig_111.savefig(path + "/figures/coloring_111.png", dpi=300, bbox_inches='tight')


