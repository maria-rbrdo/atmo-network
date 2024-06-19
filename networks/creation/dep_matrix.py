"""
This script takes the results of the SWE/TSWE simulations, splits them into
segments and finds the dependence adjacency matrices between all grid points during each
segment. This can be done using:
    - kinetic energy (task = velocity)
    - vorticity (task = vorticity)
    - height (task = height)
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

#%% PCC and MI
def PCC(data, max_lag=0, min_lag=0):
    """Find linear Pearson correlation matrix from data with (lag = True) or without
     (lag = False) lag.

     The weight magnitude of each edge is the absolute value of the maximum correlation.
     The sign of each weight indicates the direction of the interaction (E.g. C_ij>0 is from j to i)."""

    if max_lag == 0:
        return np.corrcoef(data)

    elif type(max_lag) is int and type(min_lag) is int and max_lag > 0 and min_lag >= 0:
        with (alive_bar(max_lag-min_lag, force_tty=True) as bar):

            lm = np.zeros(shape=(data.shape[0], data.shape[0]))  # lag matrix
            cm = np.zeros(shape=(data.shape[0], data.shape[0]))  # correlation matrix (NO ZERO LAG)
            bar()  # update bar

            # Positive lags — from j to i:
            for lag in range(min_lag + 1, max_lag):

                data_i = data[:, lag:]  # series i -> moving forwards
                data_j = data[:, :-lag]  # series j -> moving backwards

                cov = np.cov(data_i, data_j)  # compute covariance

                var_i = np.diag(cov[:int(len(cov) / 2), :int(len(cov) / 2)])  # variance of original series
                var_j = np.diag(cov[int(len(cov) / 2):, int(len(cov) / 2):])  # variance of lagged series
                var_ij = np.outer(var_i, var_j)  # product of variances

                corr = cov[:int(len(cov) / 2), int(len(cov) / 2):] / np.sqrt(var_ij)  # correlation (normalised var)

                mask = np.abs(cm) >= np.abs(corr)
                cm = np.where(mask, cm, corr)  # store biggest entries
                lm = np.where(mask, lm, lag * np.ones_like(lm))  # store the lag

                bar()  # update bar

        # only keep the link directions with the largest absolute value
        mask = np.abs(cm) > np.abs(cm.T)
        cm = np.where(mask, cm, np.zeros_like(cm))
        lm = np.where(mask, lm, np.zeros_like(lm))

        return cm, lm
    else:
        raise ValueError('lag must be an integer greater than zero.')


#%%
def main(model, task, method, segments, max_lag, min_lag, filename, output):
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
    file_path2 = output + f'/CM_{model}_{task}_{method}_s{segments}_l{min_lag}to{max_lag}.h5'
    try:
        os.remove(file_path2)
        print(f"Previous file '{file_path2}' deleted successfully.")
    except:
        pass

    with h5py.File(filename, mode='r') as file:

        # %% Load data
        print("Loading data...")

        dset = file['tasks'][task]

        # find coordinates
        if task == "velocity":
            theta = dset.dims[2][0][:]
            phi = dset.dims[3][0][:]
        else:
            theta = dset.dims[1][0][:]
            phi = dset.dims[2][0][:]
        t = file['scales/sim_time'][:]

        # get timeseries
        data = dset[:]  # get data
        if task == "velocity":
            u = data[:, 0, :, :]  # east-west
            v = data[:, 1, :, :]  # north-south
            data = 1 / 2 * (u ** 2 + v ** 2)  # kinetic energy
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

        # %% Calculate dependence matrix
        print("Calculating dependence matrix...")

        t_step = int(n_t / segments)  # calculate the number of timesteps on each slice
        dt = t[1]  # timestep
        max_lag = int(max_lag/dt)  # lag
        min_lag = int(min_lag/dt)  # lag

        with h5py.File(file_path1, mode='r') as ddata:
            for i in range(segments):
                # find matrix
                if method == "PCC":
                    correlation_matrix, lag_matrix = PCC(ddata["data"][:, i * t_step:(i + 1) * t_step], max_lag=max_lag,
                                                         min_lag=min_lag)
                    lag_matrix = lag_matrix * dt  # from steps to hours
                elif method == "MI":
                    print("pepe")
                else:
                    raise ValueError(f"Unknown method: {method}. Please choose from: PCC and MI.")

                # save matrix
                start_time = int(round(t[i * t_step], 3) * 1000)
                end_time = int(round(t[(i + 1) * t_step - 1], 3) * 1000)
                key = "t_" + str(start_time) + "_" + str(end_time) + "_" + str(i)

                with h5py.File(file_path2, mode='a') as store:
                    store.create_dataset(key, data=correlation_matrix)
                    store.create_dataset(key+"_lags", data=lag_matrix)

    # remove file storing intermediate data
    os.remove(file_path1)
    print(f"Previous file '{file_path1}' deleted successfully.")

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    output_path = os.path.join(args['--output'])

#    if not os.path.isdir(output_path):
#        os.mkdir(output_path)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], segments=int(args['--segments']),
#         lag=int(args['--lag']), filename=args['<files>'], output=args['--output'])

main("SWE", "vorticity", "PCC", segments=5, max_lag=24, min_lag=0,
     filename="../../data/model/SWE_snapshots/n1e5_u10_h120_m64/n1e5_u10_h120_m64_s1.h5",
     output="../../data/euler/SWE_corr/n1e5_u10_h120_m64")