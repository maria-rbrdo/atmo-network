"""
This script takes the results of the SWE/TSWE simulations, splits them into
segments and finds the vorticity adjacency matrices between all grid points during each
segment.

Data is saved to a HDF5 file in the output directory specified.

e.g.:
    $ python3 networks/creation/vort_matrix.py SWE --alpha=0
    data/model/SWE_snapshots/n1e5_u80_h120_m64/n1e5_u80_h120_m64_s1.h5 --output=data/euler/SWE_vort/n1e5_u80_h120_m64

Usage:
    vort_matrix.py <model> [--alpha=<alpha>] <files> [--output=<dir>]

Options:
    --alpha=<alpha>  Weight parameter alpha in [0,1] [default: 1/2]
    --output=<dir>  Output directory [default: ./data/euler/networks]
"""
from pyexpat import model

import numpy as np
import pandas as pd
import h5py
import os
from docopt import docopt
from alive_progress import alive_bar

def calc_distance(lat1, lat2, lon1, lon2):
    """
    Calculate the great-circle distance matrix between points specified by latitude and longitude.

    Parameters:
    lat1, lon1: Arrays of latitudes and longitudes for the first set of points.
    lat2, lon2: Arrays of latitudes and longitudes for the second set of points.

    Returns:
    A matrix of distances between each pair of points from the two sets.
    """

    R = 6371  # km radius Earth

    # Create 2D grids of latitudes and longitudes
    lat1_grid, lat2_grid = np.meshgrid(lat1, lat2)
    lon1_grid, lon2_grid = np.meshgrid(lon1, lon2)

    # Calculate differences
    Dlat = lat2_grid - lat1_grid
    Dlon = lon2_grid - lon1_grid

    # Haversine formula
    a = np.sin(Dlat / 2) ** 2 + np.cos(lat1_grid) * np.cos(lat2_grid) * np.sin(Dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - np.round(a, 15)))

    return R*c

def main(model, filename, output, alpha=0, step=1):
    """Find vorticity matrices every "step" hrs and store them."""

    # %% Prepare directory
    print('Preparing directory...')

    # delete files from previous runs
    file_path1 = output + f'/iterm_data.h5'
    try:
        os.remove(file_path1)
        print(f"Previous file '{file_path1}' deleted successfully.")
    except:
        pass
    file_path2 = output + f'/CM_{model}.h5'
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

        # %% Get grid coordinates
        print("Getting coordinates...")

        # spherical coordinates
        theta_grid = np.repeat(theta, n_phi)
        phi_grid = np.tile(phi, n_theta)

        # latitude an longitude
        lon_grid = theta_grid  # define longitude in rad
        lat_grid = (np.pi / 2 - phi_grid)  # define latitude in rad

        lon = lon_grid.reshape(-1)  # make vector
        lat = lat_grid.reshape(-1)  # make vector

        # differentials
        dt = t[1]
        dframes = int(step / dt)
        dlon = np.min(np.unique(lon_grid[lon_grid != 0]))
        dlat = np.min(np.abs(np.unique(lat_grid[lat_grid != 0])))

        # radius
        R = 6.37122e3  # km

        # save some grid coordinates
        with h5py.File(file_path2, mode='a') as store:
            store.create_dataset("theta", data=theta_grid)
            store.create_dataset("phi", data=phi_grid)

        # %% Calculate vorticity matrix
        print("Calculating vorticity matrix...")

        with h5py.File(file_path1, mode='r') as ddata:
            with alive_bar(int(n_t/dframes), force_tty=True) as bar:
                for t in range(int(n_t/dframes)):
                    # find matrix
                    vort = ddata["data"][:, t * dframes]

                    circ = np.abs(vort*R**2*np.cos(lat)*dlon*dlat)
                    circ_j, circ_i = np.meshgrid(circ, circ)

                    dist_ij = calc_distance(lat, lat, lon, lon)
                    np.fill_diagonal(dist_ij, np.inf)

                    vorticity_matrix = (alpha * circ_i + (1-alpha) * circ_j) / dist_ij

                    # save matrix
                    time = int(t * dframes * dt)
                    key = "t_" + str(time) + "_" + str(t)
                    with h5py.File(file_path2, mode='a') as store:
                        store.create_dataset(key, data=vorticity_matrix)

                    # update bar
                    bar()

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

main(model="SWE", step=72, alpha=0,
     filename="../../data/model/SWE_snapshots/n1e5_u80_h120_m64/n1e5_u80_h120_m64_s1.h5",
     output="../../data/euler/SWE_vort/n1e5_u80_h120_m64")
