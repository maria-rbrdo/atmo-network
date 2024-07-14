"""
========================================================================================================================
Vorticity-Based Adjacency Matrix Building Script
========================================================================================================================
This script reads the HDF5 net-output file from the model and finds the vorticity-based adjacency matrix between all
grid points at each time. Data is saved to a HDF5 file in the output directory specified.
------------------------------------------------------------------------------------------------------------------------
Usage:
    vort_matrix.py <files> [--dsize=<dsize>]

Options:
    --dsize=<dsize>   Down-sampling factor [default: 4]

------------------------------------------------------------------------------------------------------------------------
Notes:
- This code is written to for a spherical domain. If the domain is not spherical this network is not well built.
------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
import pandas as pd
import skimage as ski
from docopt import docopt
from alive_progress import alive_bar

#%%
def mixedterm(lat_i, lat_j, lon_i, lon_j):
    """
        This function calculates the mixed term of the velocity induced by one point source at
        another point in a sphere.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]          Description
        ------------------------------------------------------------------------------------------
        lat_i : np.array [deg]          Latitude point i.
        lat_j : np.array [deg]          Latitude point j.
        lon_i : np.array [deg]          Longitude point i.
        lon_j : np.array [deg]          Longitude point j.
        ==========================================================================================
    """

    # Transform to polar and azimuthal angles in radians
    th_i, th_j = np.pi/2 - np.deg2rad(lat_i), np.pi/2 - np.deg2rad(lat_j)
    ph_i, ph_j = np.deg2rad(lon_i), np.deg2rad(lon_j)

    # Create 2D grids of latitudes and longitudes (i -> cst in rows, j -> cst in columns)
    gth_j, gth_i = np.meshgrid(th_j, th_i)
    gph_j, gph_i = np.meshgrid(ph_j, ph_i)

    # Calculate
    c = np.cos(gth_i)*np.cos(gth_j) + np.sin(gth_i)*np.sin(gth_j)*np.cos(gph_i-gph_j)
    np.fill_diagonal(c, np.nan)

    s1 = (np.cos(gph_i)*np.sin(gph_j)*np.cos(gph_i-gph_j)-np.sin(gph_i)*np.cos(gph_j))**2
    s2 = (np.sin(gph_j)*np.sin(gph_i-gph_j))**2
    s = np.sqrt(s1+s2)
    mterm = s/(1 - c)

    return mterm

def main(qpath, hpath, dsize=4):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]          Description
        ------------------------------------------------------------------------------------------
        qpath   : string [-]            Path to the potential vorticity data file.
        hpath   : string [-]            Path to the height data file.
        dsize   : int, optional [-]     Down-sampling factor [default: 4].
        ==========================================================================================
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    print('Preparing directory...')

    info_1 = os.path.basename(qpath).split("_")
    info_2 = os.path.basename(qpath).split("_")
    assert info_1[1] == info_2[1] and info_1[2] == info_2[2], "q and h files do not cover the same time-segment."

    opath = os.path.dirname(qpath) + f'/VM_{info_1[1]}_{info_1[2]}.h5'
    try:
        os.remove(opath)
        print(f"Previous file '{opath}' deleted successfully.")
    except:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Run:
    # ------------------------------------------------------------------------------------------------------------------

    with h5py.File(qpath, mode='r') as fq:
        with h5py.File(hpath, mode='r') as fh:
            # Loading data .............................................................................................

            print("Loading data...")

            lat = fq['latitude'][:]
            lon = fq['longitude'][:]
            t = fq['time'][:]
            nlat, nlon, nt = len(lat), len(lon), len(t)

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
            R = 6.371e3  # km

            # save some grid coordinates
            with h5py.File(opath, mode='a') as store:
                store.create_dataset("latitude", data=rlat)
                store.create_dataset("longitude", data=rlon)

            # Calculating correlation matrix ...........................................................................

            print("Calculating vorticity matrix...")

            with alive_bar(int(nt), force_tty=True) as bar:
                for it in range(nt):

                    # find matrix
                    vort = fq["data"][:][:, :, it] * (fh["data"][:][:, :, it] + 10) # ASSUMED H IN KM AND AV 10 KM
                    # reduce matrix
                    rvort = ski.measure.block_reduce(vort, block_size=(dsize, dsize), func=np.mean)
                    del vort

                    # calculate vortex strength
                    gvort = rvort.reshape(-1)
                    gamma_j = np.abs(gvort*R**2*np.cos(glat)*gdlon*gdlat)
                    gamma_i = np.atleast_2d(gamma_j).T

                    # calculate mixed term
                    term_ij = mixedterm(glat, glat, glon, glon)
                    np.fill_diagonal(term_ij, 0)

                    # calculate adjacency matrix
                    u_jti = 1/(4*np.pi*R) * (0 * gamma_i + 1 * gamma_j) * term_ij
                    mvort = np.abs(u_jti)

                    # save matrix
                    key = "t_" + str(t[it]) + "_" + str(it)
                    with h5py.File(opath, mode='a') as store:
                        store.create_dataset(key, data=mvort)

                    # update bar
                    bar()

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    output_path = os.path.join(args['--output'])

#    if not os.path.isdir(output_path):
#        os.mkdir(output_path)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], segments=int(args['--segments']),
#         lag=int(args['--lag']), filename=args['<files>'], output=args['--output'])

ss = [600]
for s in ss:
    main(f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/q_1763_1803",
         f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/h_1763_1803")