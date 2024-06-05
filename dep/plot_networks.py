import os
import h5py
import numpy as np
from alive_progress import alive_bar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from network_properties import calc_density

def plot(indep_var, indep_var_name, dep_var, dep_var_name, std):
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    df = pd.DataFrame({'x': indep_var, 'y': dep_var})
    sns.lineplot(df, ax=ax, x='x', y='y', marker='o', markersize=5, color="black")
    ax.errorbar(indep_var, dep_var, yerr=std, fmt='none', capsize=5, zorder=1, color='black')

    ax.set_xlabel(f'{dep_var_name}')
    ax.set_ylabel(f'{indep_var_name}')
def main(filenames, indep_var, indep_var_name):
    with alive_bar(len(filenames), force_tty=True) as bar:
        for filename in filenames:
            f_densities_mean = []
            f_densities_std = []
            f_filenames = []
            with h5py.File(filename, mode='r') as f:
                density_mean = []
                density_std = []
                for k in set(f.keys()) - {"theta", "phi"}:
                    # load
                    cm = f[k][:]
                    # calculate
                    d_mean, d_std = calc_density(cm)  # calculate density
                    # append:
                    density_mean.append(d_mean)
                    density_std.append(d_std)
                # average and std of each file:
                f_densities_mean.append(np.mean(density_mean))
                f_densities_std.append(np.mean(density_std))
                f_filenames.append(filename)
            plot(indep_var, indep_var_name, f_densities_mean, "average density", f_densities_std)
            bar()

main([], [0, 0.25, 0.5, 0.75, 0.9], "threshold")