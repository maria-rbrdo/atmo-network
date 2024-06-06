import os
import h5py
import numpy as np
from alive_progress import alive_bar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from network_properties import *

def plot_line(savename, df, dpi=200):

    fig, ax = plt.subplots(figsize=(20, 7))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    sns.lineplot(df, ax=ax, x=df.columns[0], y=df.columns[1], marker='o', markersize=5, color="black")

    ax.set_xlabel(f'{df.columns[0]}')
    ax.set_ylabel(f'{df.columns[1]}')

    ax.set_yscale('log')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    fig.clear()

def plot_hist(savename, df, dpi=200):
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    sns.histplot(df, ax=ax, x=df.columns[1], hue=df.columns[0], element="step", stat="percent", common_norm=False,
                 palette=sns.color_palette(), bins=50)

    ax.set_xlabel(f'{df.columns[1]}')

    # save figure
    fig.savefig(savename, dpi=dpi, bbox_inches='tight')
    fig.clear()
def main(filename, output, indep_var, indep_var_name, dep_var_name):
    with h5py.File(filename, mode='r') as f:
        theta = f["theta"][:]
        phi = f["phi"][:]
        lon = (np.pi - theta)
        lat = (np.pi / 2 - phi)
        df = pd.DataFrame(columns=[indep_var_name, dep_var_name])
        for val in indep_var:
            print(f"{val}:")
            for k in set(f.keys()) - {"theta", "phi"}:
                # load
                cm = f[k][:]  # get correlation data
                np.fill_diagonal(cm, 0)  # take out diagonal
                cm[np.abs(cm) <= val] = 0  # impose threshold
                cm = np.where(cm > 0, 1, np.where(cm < 0, -1, 0))  # unweighted matrix
                # calculate
                values = calc_distances(cm, lon, lat)  # calculate density
                # append
                new_rows = pd.DataFrame({indep_var_name: [val]*len(values), dep_var_name: values})
                df = pd.concat([df, new_rows], ignore_index=True)
        #plot
        print("Creating line plot...")
        plot_line(output+"line.png", df)
        print("Creating histogram plot...")
        plot_hist(output+"hist.png", df)

main("../data/euler/SWE_corr/CM_SWE_velocity_PCC_s1_l24.h5", "../data/euler/SWE_corr/",
     [0.97, 0.99], "threshold", "distance (km)")
