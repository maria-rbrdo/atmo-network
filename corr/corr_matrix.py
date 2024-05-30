"""
This script takes the results of the SWE/TSWE simulations and finds the
corr matrix (using Pearson corr) between all grid points.
Data is saved to a HDF5 file in the output directory specified.

e.g.:
    $ python3 corr/corr_matrix.py SWE velocity --segments=15 data/model/SWE_snapshots/SWE_snapshots_s1.h5 --output=data/euler/SWE_corr

Usage:
    corr_matrix.py <model> <task> [--segments=<seg>] <files> [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/corr]
    --segments=<seg>  Segments in which to break time series [default: 1]

"""

import numpy as np
import h5py
import os
from docopt import docopt
from alive_progress import alive_bar

def main(filename, output, model, task, segments):
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

        #%% Load data
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
            data = 1/2 * (u**2 + v**2)  # kinetic energy
        n_t, n_theta, n_phi = data.shape

        #%% Get coordinates
        print("Getting coordinates...")
        theta_indices = np.repeat(theta, n_phi)
        phi_indices = np.tile(phi, n_theta)
        with h5py.File(output + f'/CM_{model}_{task}.h5', mode='a') as store:
            store.create_dataset("theta", data=theta_indices)
            store.create_dataset("phi", data=phi_indices)

        #%% Calculate correlation matrix
        print("Calculating correlation matrix...")

        ddata = data.reshape(n_t, -1).T
        dt = int(n_t/segments)
        with h5py.File(output + f'/CM_{model}_{task}_data.h5', mode='a') as store:
            store.create_dataset("data", data=ddata)

        del data, ddata

        with alive_bar(segments) as bar:
            with h5py.File(output + f'/CM_{model}_{task}_data.h5', mode='r') as ddata:
                for i in range(segments):
                    # find matrix
                    correlation_matrix = np.corrcoef(ddata["data"][:, i*dt:(i+1)*dt])

                    # save matrix
                    start_time = int(round(t[i*dt], 3) * 1000)
                    end_time = int(round(t[(i+1)*dt-1], 3) * 1000)
                    key = "t_"+str(start_time)+"_"+str(end_time)+"_"+str(i)
                    with h5py.File(output + f'/CM_{model}_{task}.h5', mode='a') as store:
                        store.create_dataset(key, data=correlation_matrix)

                    # update bar
                    bar()

    # remove file storing intermediate data
    os.remove(file_path1)
    print(f"Previous file '{file_path1}' deleted successfully.")

if __name__ == "__main__":

    args = docopt(__doc__)

    main(filename=args['<files>'], output=args['--output'], model=args['<model>'], task=args['<task>'],
         segments=int(args['--segments']))
