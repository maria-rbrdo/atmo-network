import h5py
import numpy as np
import scipy as sp

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from alive_progress import alive_bar

def find_split(A, n):

    D = np.diag(np.sum(A, axis=1))
    #Dinv = np.diag(1/np.sum(A, axis=1))
    L = D - A
    #DinvL = Dinv @ L

    # eigenvalues
    #l1, x1 = sp.linalg.eigs(DinvL, subset_by_index=[A.shape[0] - n, A.shape[0] - 1])
    l, x = sp.linalg.eigh(L, D, subset_by_index=[A.shape[0] - n, A.shape[0] - 1])
    x = np.flip(x, axis=1)  # decreasing order

    # split
    s = np.sign(x - np.mean(x, axis=0)).astype(int)
    s[s == 0] = 1 if np.random.rand() < 0.5 else -1
    s[s == -1] = 0

    # from binary to int
    s = np.array([''.join(s[i, :].astype('str')) for i in range(A.shape[0])])

    return s

def create_subplot(fig, position, title, projection=ccrs.Orthographic(0, 60), sph=True):
    ax = fig.add_subplot(position[0], position[1], position[2], projection=projection)
    if sph:
        ax.set_global()
        ax.gridlines(color="k", linestyle=':')
    ax.set_title(title)
    return ax

sph = False
#with h5py.File("../../../dataloc/pv50-nu4-urlx.c0sat600.T170/netdata/TM_1280_1300.h5", mode='r') as f:
with h5py.File("../../../dataloc/quadgyre/netdata/TM.h5", mode='r') as f:
    print("Getting coordinates...")
    coord = f["coord"][:]

    print("Clustering...")
    s = find_split(f["adj"][:], 3)

    print("Plotting...")
    # Initialize the figure
    fig = plt.figure(figsize=(20, 10))

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

    color_dict = {"000": "brown", "001": "red", "010": "orange", "011": "gold",
                  "100": "navy", "101": "blueviolet", "110": "cyan", "111": "royalblue"}

    with alive_bar(coord.shape[0], force_tty=True) as bar:
        for n in range(coord.shape[0]):
            branch = s[n]
            color = color_dict[branch]

            ax_list = [ax_all]
            if branch[0] == "0":
                ax_list.append(ax_0)
                if branch[1] == "0":
                    ax_list.append(ax_00)
                    if branch[2] == "0":
                        ax_list.append(ax_000)
                    else:
                        ax_list.append(ax_001)
                else:
                    ax_list.append(ax_01)
                    if branch[2] == "0":
                        ax_list.append(ax_010)
                    else:
                        ax_list.append(ax_011)
            else:
                ax_list.append(ax_1)
                if branch[1] == "0":
                    ax_list.append(ax_10)
                    if branch[2] == "0":
                        ax_list.append(ax_100)
                    else:
                        ax_list.append(ax_101)
                else:
                    ax_list.append(ax_11)
                    if branch[2] == "0":
                        ax_list.append(ax_110)
                    else:
                        ax_list.append(ax_111)

            for ax in ax_list:
                if sph:
                    ax.scatter(coord[n, 1, 0], coord[n, 0, 0], c=color, marker="D", s=0.5, transform=ccrs.Geodetic())
                    ax.plot(coord[n, 1, :], coord[n, 0, :], "-", color=color, alpha=0.10, transform=ccrs.Geodetic())
                else:
                    ax.scatter(coord[n, 1, 0], coord[n, 0, 0], c=color, marker="D", s=0.5)
                    ax.plot(coord[n, 1, :], coord[n, 0, :], "-", color=color, alpha=0.10)
            bar()
    plt.tight_layout()
    plt.show()


