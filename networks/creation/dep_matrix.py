"""
========================================================================================================================
Correlation-Based Adjacency Matrix Building Script
========================================================================================================================
This script reads the HDF5 net-output file from the model, splits it into segments, and finds the correlation-based
adjacency matrix between all grid points for each segment.

Correlations can be found at zero lag (lag = False) or at time lag (lag = True). Data is saved to a HDF5 file in the
output directory specified.
------------------------------------------------------------------------------------------------------------------------
Usage:
    dep_matrix.py <files> <lmax> [--lmin=<lmin>] [--segments=<segments>]

Options:
    --output=<dir>  Output directory [default: ./data/euler/networks]
    --segments=<seg>  Segments in which to break time series [default: 1]
    --lag=<lag>  Maximum lag considered [default: 0]
------------------------------------------------------------------------------------------------------------------------
Notes:
- Available fields: ('h', 'd', 'q', 'z')
------------------------------------------------------------------------------------------------------------------------
"""
from pyexpat import model

import numpy as np
import pandas as pd
import h5py
import os
from docopt import docopt
from alive_progress import alive_bar

#%% PCC and MI
def PCC(data, max_lag, min_lag=0):
    """
        This function finds the Pearson correlation matrix from data. The
        weight magnitude of each edge is the absolute value of the maximum
        correlation.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type              Description
        ------------------------------------------------------------------------------------------
        data    : np.array          Array of data where each row is the timeseries of a node.
        max_lag : int               Maximum lag considered
        min_lag : int, optional     Minimum lag considered [default: 0].
        ==========================================================================================

        Returns:
        -------
        numpy.ndarray
            Adjacency matrix, with shape (nlon*nlat, nlon*nlat).
        """

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
def main(fpath, lmax, lmin=0, segments=1):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]          Description
        ------------------------------------------------------------------------------------------
        fpath   : string [-]            Path to the data file.
        lmax : int [days]               Maximum lag considered.
        lmin : int, optional [days]     Minimum lag considered [default: 0].
        segments: int, optional [-]     Segments in which to break time series [default: 1].
        ==========================================================================================
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    print('Preparing directory...')

    # obtain data from fpath
    fld = os.path.basename(fpath).split("_")[0]
    opath = os.path.dirname(fpath) + f'/CM_{model}_{fld}_s{segments}_l{lmin}to{lmax}.h5'

    # delete files from previous runs
    try:
        os.remove(opath)
        print(f"Previous file '{opath}' deleted successfully.")
    except:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Run:
    # ------------------------------------------------------------------------------------------------------------------

    with h5py.File(fpath, mode='r') as f:

        # Loading data .................................................................................................

        print("Loading data...")

        lat = f['latitude'][:]
        lon = f['longitude'][:]
        t = f['time'][:]

        nlat, nlon, nt = len(lat), len(lon), len(t)

        diter = int(nt / segments)  # iterations/slice
        dt = t[1] - t[0]  # time-step
        lmax = int(lmax/dt)  # max lag in iter
        lmin = int(lmin/dt)  # min lag in iter

        with h5py.File(opath, mode='a') as store:
            mlat = np.tile(lat, nlon)  # mesh latitude
            mlon = np.repeat(lon, nlat)  # mesh longitude
            store.create_dataset("latitude", data=mlat)
            store.create_dataset("longitude", data=mlon)

        # Calculating correlation matrix ...............................................................................

        print("Calculating correlation matrix...")

        for i in range(segments):

            cdata = f["data"][:, :, i*diter:(i+1)*diter].reshape(-1, nt)
            mcorr, mlag = PCC(cdata, max_lag=lmax, min_lag=lmin)
            mlag = mlag * dt  # from steps to hours

            # save matrix
            tstart = int(round(t[i * diter], 3) * 1000)
            tend = int(round(t[(i + 1)*diter-1], 3) * 1000)
            key = "t_" + str(tstart) + "_" + str(tend) + "_" + str(i)

            with h5py.File(opath, mode='a') as store:
                store.create_dataset(key, data=mcorr)
                store.create_dataset(key+"_lags", data=mlag)

#%%
if __name__ == "__main__":
    args = docopt(__doc__)
    main(fpath=args['<files>'], lmax=int(args['<lmax>']), lmin=int(args['--lmin']), segments=int(args['--segments']))
