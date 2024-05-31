"""
This script takes the results of the SWE/TSWE simulations and finds the
corr matrix (using Pearson corr) between all grid points.
Data is saved to a HDF5 file in the output directory specified.

e.g.:
    $ python3 corr/dep_matrix.py SWE velocity PCC --segments=15 --lagged=True
        data/model/SWE_snapshots/SWE_snapshots_s1.h5 --output=data/euler/SWE_corr

Usage:
    dep_matrix.py <model> <task> <method> [--segments=<seg>] [--lag=<lag>] <files> [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/corr]
    --segments=<seg>  Segments in which to break time series [default: 1]
    --lag=<lag>  Maximum lag considered [default: 0]

"""

import numpy as np
from scipy.ndimage import shift
import h5py
import os
from docopt import docopt
from alive_progress import alive_bar


def PCC(data, lag):
    if lag == 0:
        return np.corrcoef(data)
    elif lag > 0:
        cm = np.zeros(shape=(len(data), len(data)))
        np.fill_diagonal(cm, 1)
        with alive_bar(len(data), force_tty=True) as bar:
            for i in range(len(data)):
                x_i = np.vstack([shift(data[i, :], l, cval=np.NaN) for l in range(lag)])
                for j in range(i + 1, len(data)):
                    x_j = data[j, :]
                    max_corr = 0
                    for l in range(lag):
                        cov = np.cov(np.vstack([x_i[l, :], x_j]))
                        corr = cov[1, 0] / np.sqrt(cov[0, 0] * cov[1, 1])
                        if corr > max_corr:
                            max_corr = corr
                    cm[i, j] = max_corr
                    cm[j, i] = max_corr
                bar()
        return cm
    else:
        raise ValueError(f"Invalid value for lag: {lag} (should be a positive number)")


# def MI(lagged):

def main(model, task, method, segments, lag, filename, output):
    # %% Prepare directory
    print('Preparing directory...')

    # delete files from previous runs
    file_path1 = output + f'/CM_{model}_{task}_data.h5'
    try:
        os.remove(file_path1)
        print(f"Previous file '{file_path1}' deleted successfully.")
    except:
        pass
    file_path2 = output + f'/CM_{model}_{task}.h5'
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
        theta = dset.dims[2][0][:]
        phi = dset.dims[3][0][:]
        t = file['scales/sim_time'][:]

        # get timeseries
        data = dset[:]  # get data
        if task == "velocity":
            u = data[:, 0, :, :]  # east-west
            v = data[:, 1, :, :]  # north-south
            data = 1 / 2 * (u ** 2 + v ** 2)  # kinetic energy
        n_t, n_theta, n_phi = data.shape

        # store data
        ddata = data.reshape(n_t, -1).T
        with h5py.File(output + f'/CM_{model}_{task}_data.h5', mode='a') as store:
            store.create_dataset("data", data=ddata)
        del data, ddata

        # %% Get coordinates
        print("Getting coordinates...")
        theta_indices = np.repeat(theta, n_phi)
        phi_indices = np.tile(phi, n_theta)
        with h5py.File(output + f'/CM_{model}_{task}.h5', mode='a') as store:
            store.create_dataset("theta", data=theta_indices)
            store.create_dataset("phi", data=phi_indices)

        # %% Calculate correlation matrix
        print("Calculating correlation matrix...")

        t_step = int(n_t / segments)
        dt = t[1]  # timestep
        lag = int(lag / dt)  # lag in steps instead of time
        with h5py.File(output + f'/CM_{model}_{task}_data.h5', mode='r') as ddata:
            for i in range(segments):
                # find matrix
                if method == "PCC":
                    correlation_matrix = PCC(ddata["data"][:, i * t_step:(i + 1) * t_step], lag)
                    print("hi")
                elif method == "MI":
                    print("pepe")
                else:
                    raise ValueError(f"Unknown method: {method}. Please choose from: PCC and MI.")

                # save matrix
                start_time = int(round(t[i * t_step], 3) * 1000)
                end_time = int(round(t[(i + 1) * t_step - 1], 3) * 1000)
                key = "t_" + str(start_time) + "_" + str(end_time) + "_" + str(i)
                with h5py.File(output + f'/CM_{model}_{task}.h5', mode='a') as store:
                    store.create_dataset(key, data=correlation_matrix)

    # remove file storing intermediate data
    os.remove(file_path1)
    print(f"Previous file '{file_path1}' deleted successfully.")

main("SWE", "velocity", "PCC", segments=1, lag=1,
     filename="../data/model/SWE_snapshots/SWE_snapshots_s1.h5", output="../data/euler/SWE_corr")


#if __name__ == "__main__":
#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], segments=int(args['--segments']),
#         lag=float(args['--lag']), filename=args['<files>'], output=args['--output'])
