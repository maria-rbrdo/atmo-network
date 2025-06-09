"""
====================================================================================================
Vorticity-Based Adjacency Matrix Building Script
====================================================================================================
This script reads the HDF5 net-output file from the model and finds the vorticity-based adjacency
matrix between all grid points at each time. Data is saved to a HDF5 file in the output directory
specified.
----------------------------------------------------------------------------------------------------
Notes:
-This code is written to for a spherical domain. If the domain is not spherical this network is not
well-built.
----------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
import skimage as ski

# -------------------------------------------------------------------------------------------------
# Functions:
# -------------------------------------------------------------------------------------------------

def mterm(lat_i, lat_j, lon_i, lon_j):
    """
        This function calculates the mixed term of the velocity induced by one point source at
        another point in a sphere.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]          Description
        ------------------------------------------------------------------------------------------
        lat_i : numpy.ndarray [deg]     Latitude point i.
        lat_j : numpy.ndarray [deg]     Latitude point j.
        lon_i : numpy.ndarray [deg]     Longitude point i.
        lon_j : numpy.ndarray [deg]     Longitude point j.
        ==========================================================================================
    """

    # Transform to polar and azimuthal angles in radians
    th_i, th_j = np.pi/2 - np.deg2rad(lat_i), np.pi/2 - np.deg2rad(lat_j)
    ph_i, ph_j = np.deg2rad(lon_i), np.deg2rad(lon_j)

    # Create 2D grids of latitudes and longitudes (i -> cst in rows, j -> cst in columns)
    gth_j, gth_i = np.meshgrid(th_j, th_i)
    gph_j, gph_i = np.meshgrid(ph_j, ph_i)

    # Calculate
    num1 = (np.cos(gth_i)*np.sin(gth_j)*np.cos(gph_i-gph_j)-np.sin(gth_i)*np.cos(gth_j))**2
    num2 = (np.sin(gth_j)*np.sin(gph_i-gph_j))**2
    num = np.sqrt(num1+num2)

    denom = 1 - (np.cos(gth_i)*np.cos(gth_j) + np.sin(gth_i)*np.sin(gth_j)*np.cos(gph_i-gph_j))
    np.fill_diagonal(denom, np.inf)

    mterm = num/denom

    return mterm

# -------------------------------------------------------------------------------------------------
# Main:
# -------------------------------------------------------------------------------------------------

def main(fpath, dsize=4):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]          Description
        ------------------------------------------------------------------------------------------
        fpath   : string [-]            Path to the relative vorticity data file.
        dsize   : int, optional [-]     Down-sampling factor [default: 4].
        ==========================================================================================
    """

    # Prepare directory :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    print('Preparing directory...')

    info = os.path.basename(fpath).split("_")

    try:
        opath = os.path.dirname(fpath) + f'/VM_{info[1]}_{info[2]}.h5'
    except:
        opath = os.path.dirname(fpath) + f'/VM.h5'

    try:
        os.remove(opath)
        print(f"Previous file '{opath}' deleted successfully.")
    except:
        pass

    # Run :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    with h5py.File(fpath, mode='r') as f:

        # Loading data ............................................................................

        print("Loading data...")

        lat = f['latitude'][:]
        lon = f['longitude'][:]
        t = f['time'][:]
        nlat, nlon, nt = len(lat), len(lon), len(t)
        dt = t[1] - t[0]

        # reduced latitude and longitude
        rlat = ski.measure.block_reduce(lat, block_size=dsize, func=np.mean)
        rlon = ski.measure.block_reduce(lon, block_size=dsize, func=np.mean)

        # grid
        glon, glat = np.meshgrid(rlon, rlat)
        glon, glat = glon.reshape(-1), glat.reshape(-1)

        # space between grid points
        ddlat = rlat[:-1] - rlat[1:]
        ddlon = rlon[1:] - rlon[:-1]

        # sice of grid point cell
        dlat = np.hstack((ddlat[0], 1 / 2 * (ddlat[1:] + ddlat[:-1]), np.atleast_1d(ddlat[-1])))
        dlon = np.hstack((ddlon[0], 1 / 2 * (ddlon[1:] + ddlon[:-1]), np.atleast_1d(ddlon[-1])))

        # grid
        gdlon, gdlat = np.meshgrid(dlon, dlat)
        gdlon, gdlat = gdlon.reshape(-1), gdlat.reshape(-1)

        # delete unnecessary arrays
        del lat, lon, ddlat, ddlon

        # Earth's radius
        R = 6.371e6  # m

        # save some grid coordinates
        with h5py.File(opath, mode='a') as store:
            store.create_dataset("latitude", data=rlat)
            store.create_dataset("longitude", data=rlon)

        # Calculating matrix ......................................................................

        print("Calculating vorticity matrix...")
        for it in tqdm(range(nt)):
            # find matrix
            vort = f["vorticity"][:][:, :, it]

            # reduce matrix
            rvort = ski.measure.block_reduce(vort, block_size=(dsize, dsize), func=np.mean)
            del vort

            # calculate vortex strength
            gvort = rvort.reshape(-1)
            gamma_j = np.abs(gvort*R**2*np.cos(np.deg2rad(glat))*gdlon*gdlat)
            gamma_i = np.atleast_2d(gamma_j).T

            # calculate mixed term
            term = mterm(glat, glat, glon, glon)

            # calculate adjacency matrix
            u_jti = 1/(4*np.pi*R) * (0 * gamma_i + 1 * gamma_j) * term
            mvort = np.abs(u_jti)

            # save matrix
            key = "t_" + str(t[it]) + "_" + str(it)
            with h5py.File(opath, mode='a') as store:
                store.create_dataset(key, data=mvort)

            del rvort, gvort, gamma_j, gamma_i, term, u_jti, mvort

# -------------------------------------------------------------------------------------------------
# Run:
# -------------------------------------------------------------------------------------------------

main(f"~/z", dsize=2)