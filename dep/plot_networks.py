import os
import h5py
import numpy as np
from alive_progress import alive_bar
from docopt import docopt
from network_properties import calc_density

def main(filenames):

    for filename in filenames:
        f_densities_mean = []
        f_densities_std = []
        f_filenames = []
        with h5py.File(filename, mode='r') as f:
            density_mean = []
            density_std = []
            for k in set(f.keys()) - {"theta", "phi"}:
                #%% Load data
                cm = f[k][:]  # get correlation data
                d_mean, d_std = calc_density(cm)
                density_mean.append(d_mean)
                density_std.append(d_std)
            f_densities_mean.append(np.mean(density_mean))
            f_densities_std.append(np.mean(density_std))
