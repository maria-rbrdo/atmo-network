"""
====================================================================================================
Correlation-Based Adjacency Matrix Building Script
====================================================================================================
This script reads the HDF5 net-output file from the model, splits it into segments, and finds the 
correlation-based adjacency matrix between all grid points for each segment.

Correlations can be found at zero lag (lag = False) or at time lag (lag = True). Data is saved to a 
HDF5 file in the output directory specified.
----------------------------------------------------------------------------------------------------
"""

import os

import h5py
import numpy as np
import skimage as ski
from alive_progress import alive_bar


# %%
def pcc(data, max_lag, min_lag=0):
    """
    This function finds the Pearson correlation matrix from data. The weight magnitude of each edge
    is the absolute value of the maximum correlation.

    ================================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------------
    Name    : Type              Description
    ------------------------------------------------------------------------------------------------
    data    : numpy.ndarray     Array of data where each row is the timeseries of a node.
    max_lag : int               Maximum lag considered
    min_lag : int, optional     Minimum lag considered [default: 0].
    ================================================================================================

    Returns:
    -------
    numpy.ndarray
        Adjacency matrix, with shape (nlon*nlat, nlon*nlat).
    numpy.ndarray
        Matrix containing the time-lag of the link.
    """

    if max_lag == 0:
        return np.corrcoef(data), np.zeros(shape=(data.shape[0], data.shape[0]))

    elif (
        isinstance(max_lag, int) and isinstance(min_lag, int) and 0 <= min_lag < max_lag
    ):
        with alive_bar(max_lag - min_lag - 1, force_tty=True) as bar:

            lm = np.zeros(shape=(data.shape[0], data.shape[0]))  # lag matrix
            cm = np.zeros(shape=(data.shape[0], data.shape[0]))  # correlation matrix

            # Positive lags — from j to i:
            for lag in range(min_lag, max_lag):

                if lag == 0:
                    continue  # not considering instantaneous links

                # Lag timeseries
                data_i = data[:, lag:]  # series i -> moving forwards (i.e. orig)
                data_j = data[:, :-lag]  # series j -> moving backwards (i.e. lagged)

                # Correlation
                corr = np.corrcoef(data_i, data_j)
                corr = corr[
                    : corr.shape[0] // 2, corr.shape[0] // 2 :
                ]  # entries of interest (ivsj)

                # Store info
                mask = np.abs(cm) >= np.abs(corr)
                cm = np.where(mask, cm, corr)  # store biggest entries
                lm = np.where(mask, lm, lag * np.ones_like(lm))  # store the lag

                bar()  # update bar

        # only keep the link directions with the largest absolute value
        mask = np.abs(cm) > np.abs(cm.T)
        cm = np.where(mask, cm, np.zeros_like(cm))
        lm = np.where(mask, lm, np.zeros_like(lm))

        return np.abs(cm), lm

    else:
        raise ValueError(
            "max_lag and min_lag must be integers fulfilling 0<=min_lag<max_lag"
        )


# %%
def main(fpath, fld, lmax, lmin=0, window_size=1, window_step=1, dsize=2):
    """
    ================================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------------
    Name    : Type [units]                  Description
    ------------------------------------------------------------------------------------------------
    fpath   : string [-]                    Path to the data file.
    fld : string [-]                        Field to study (e.g "pv")
    lmax : int [days]                       Maximum lag considered.
    lmin : int, optional [days]             Minimum lag considered [default: 0].
    window_size: float, optional [int]      Sliding window size [default: 1].
    window_step: float, optional [int]      Sliding window step [default: 1].
    dsize: int, optional [-]                Down-sampling factor [default: 4].
    ================================================================================================
    """

    # ----------------------------------------------------------------------------------------------
    # Prepare directory:
    # ----------------------------------------------------------------------------------------------

    print("Preparing directory...")

    # obtain data from fpath
    info = os.path.basename(fpath).split(".")[0]
    opath = (
        os.path.dirname(fpath)
        + f"/CM-{fld}-{info}-{window_size}x{window_step}-{lmin}to{lmax}.h5"
    )

    # delete files from previous runs
    try:
        os.remove(opath)
        print(f"Previous file '{opath}' deleted successfully.")
    except OSError:
        pass

    # ----------------------------------------------------------------------------------------------
    # Run:
    # ----------------------------------------------------------------------------------------------

    with (h5py.File(fpath, mode="r") as f):

        # Loading data .............................................................................

        print("Loading data...")

        # load data
        try:
            lat = np.array(f["latitude"])
            lon = np.array(f["longitude"])
            t = np.array(f["time"])
        except KeyError:
            print("File must include latitude, longitude, and time data")
            raise

        # reduced latitude and longitude
        rlat = ski.measure.block_reduce(lat, block_size=dsize, func=np.mean)
        rlon = ski.measure.block_reduce(lon, block_size=dsize, func=np.mean)

        # delete unnecessary arrays
        del lat, lon

        # time-step and number of steps
        dt = t[1] - t[0]
        nt = len(t)

        # other
        isize = int(window_size / dt)  # iterations/slice
        istep = int(window_step / dt)  # iterations step
        if istep != 0:
            nsteps = int((nt - isize) / istep + 1)  # n steps
        else:
            nsteps = 1
        lmax = int(lmax / dt)  # max lag in iter
        lmin = int(lmin / dt)  # min lag in iter

        print(f"* time segments cover: {(nsteps-1)*istep*dt+isize*dt} days")

        # store grid data
        with h5py.File(opath, mode="a") as store:
            store.create_dataset("latitude", data=rlat)
            store.create_dataset("longitude", data=rlon)

        # Calculating correlation matrix ...........................................................

        print("Calculating correlation matrix...")

        for i in range(nsteps):
            tstart = t[i * istep]
            tend = t[i * istep + isize - 1]
            print(f"* segment ({tstart}-{tend} days): {i+1}/{nsteps}")

            # find matrix
            data = np.array(f[fld])[:, :, i * istep : i * istep + isize - 1]

            # reduce matrix
            rdata = ski.measure.block_reduce(
                data, block_size=(dsize, dsize, 1), func=np.mean
            )

            # squeeze matrix
            rsdata = rdata.reshape(-1, isize - 1)

            # standarise
            ddata = (rsdata - np.mean(rsdata, axis=1).reshape(-1, 1))
            ddata /= np.std(rsdata, axis=1).reshape(-1, 1)

            del data, rdata, rsdata

            # calculate adjacency matrix
            mcorr, mlag = pcc(ddata, max_lag=lmax, min_lag=lmin)
            mlag = mlag * dt  # from steps to hours

            # save adjacency matrix
            key = "t_" + str(tstart) + "_" + str(tend) + "_" + str(i)
            with h5py.File(opath, mode="a") as store:
                store.create_dataset(key, data=mcorr)
                store.create_dataset(key + "_lags", data=mlag)

main(fpath="../../../../output/ERA5/Y2000-DJFM-daily-NH-850K/data/Y2000-DJFM-daily-NH-850K.h5",
     fld="pv", lmax=0, lmin=0, window_size=30, window_step=30, dsize=1)