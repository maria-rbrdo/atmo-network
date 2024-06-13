"""
This script takes the results of the SWE/TSWE simulations, splits them into
segments and finds the dependence adjacency matrices between all grid points during each
segment. This can be done using:
    - kinetic energy (task = velocity)
The available methods are:
    - linear correlation (method = PCC)
    - mutual information (method = MI)
which can be found at zero lag (lag = False) or with time lag (lag = True). Data is
saved to a HDF5 file in the output directory specified.

e.g.:
    $ python3 networks/creation/dep_matrix.py SWE velocity PCC --segments=1 --lag=24 data/model/SWE_snapshots/SWE_snapshots_s1.h5 --output=data/euler/SWE_corr

Usage:
    dep_matrix.py <model> <task> <method> [--segments=<seg>] [--lag=<lag>] <files> [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/networks]
    --segments=<seg>  Segments in which to break time series [default: 1]
    --lag=<lag>  Maximum lag considered [default: 0]
"""
from pyexpat import model

import numpy as np
import pandas as pd
import h5py
import os
from docopt import docopt
from alive_progress import alive_bar

def main(model, filename, output):
    """Find dependence matrices and store them."""

    # %% Prepare directory
    print('Preparing directory...')

    # delete files from previous runs
    file_path1 = output + f'/iterm_data.h5'
    try:
        os.remove(file_path1)
        print(f"Previous file '{file_path1}' deleted successfully.")
    except:
        pass
    file_path2 = output + f'/CM_{model}_{task}_{method}_s{segments}_l{lag}.h5'
    try:
        os.remove(file_path2)
        print(f"Previous file '{file_path2}' deleted successfully.")
    except:
        pass

    with h5py.File(filename, mode='r') as file:

        # %% Load data
        print("Loading data...")

        dset = file['tasks']['vorticity']

        # find coordinates
        theta = dset.dims[1][0][:]
        phi = dset.dims[2][0][:]
        t = file['scales/sim_time'][:]

        # get timeseries
        data = dset[:]  # get data
        n_t, n_theta, n_phi = data.shape

        # keep data in a file to save RAM
        ddata = data.reshape(n_t, -1).T
        with h5py.File(file_path1, mode='a') as store:
            store.create_dataset("data", data=ddata)
        del data, ddata

        # %% Get coordinates
        print("Getting coordinates...")
        theta_indices = np.repeat(theta, n_phi)
        phi_indices = np.tile(phi, n_theta)

        # save coordinates
        with h5py.File(file_path2, mode='a') as store:
            store.create_dataset("theta", data=theta_indices)
            store.create_dataset("phi", data=phi_indices)

        # %% Calculate vorticity matrix
        print("Calculating vorticity matrix...")

        with h5py.File(file_path1, mode='r') as ddata:
            for i in range(n_t):
                # find matrix


                # save matrix
                start_time = int(round(t[i * t_step], 3) * 1000)
                end_time = int(round(t[(i + 1) * t_step - 1], 3) * 1000)
                key = "t_" + str(start_time) + "_" + str(end_time) + "_" + str(i)
                with h5py.File(file_path2, mode='a') as store:
                    store.create_dataset(key, data=correlation_matrix)

    # remove file storing intermediate data
    os.remove(file_path1)
    print(f"Previous file '{file_path1}' deleted successfully.")

if __name__ == "__main__":

    args = docopt(__doc__)

    output_path = os.path.join(args['--output'])

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], segments=int(args['--segments']),
         lag=int(args['--lag']), filename=args['<files>'], output=args['--output'])

#main("SWE", "vorticity", "PCC", segments=15, lag=23,
#     filename="../../data/model/SWE_snapshots/SWE_snapshots_s1.h5", output="../../data/euler/SWE_corr")
