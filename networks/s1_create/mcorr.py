"""
========================================================================================================================
Correlation-Based Adjacency Matrix Building Script
========================================================================================================================
This script reads the HDF5 net-output file from the model, splits it into segments, and finds the correlation-based
adjacency matrix between all grid points for each segment.

Correlations can be found at zero lag (lag = False) or at time lag (lag = True). Data is saved to a HDF5 file in the
output directory specified.
------------------------------------------------------------------------------------------------------------------------
"""
import os
import h5py
import numpy as np
import skimage as ski
from alive_progress import alive_bar

#%%
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
        data    : numpy.ndarray     Array of data where each row is the timeseries of a node.
        max_lag : int               Maximum lag considered
        min_lag : int, optional     Minimum lag considered [default: 0].
        ==========================================================================================

        Returns:
        -------
        numpy.ndarray
            Adjacency matrix, with shape (nlon*nlat, nlon*nlat).
        """

    if max_lag == 0:
        return np.corrcoef(data), np.zeros(shape=(data.shape[0], data.shape[0]))

    elif type(max_lag) is int and type(min_lag) is int and max_lag > 0 and min_lag >= 0:
        with alive_bar(max_lag-min_lag-1, force_tty=True) as bar:

            lm = np.zeros(shape=(data.shape[0], data.shape[0]))  # lag matrix
            cm = np.zeros(shape=(data.shape[0], data.shape[0]))  # correlation matrix (NO ZERO LAG)

            # Positive lags — from j to i:
            for lag in range(min_lag+1, max_lag):

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
def main(fpath, lmax, lmin=0, window_size=1, window_step=1, dsize=2):
    """
        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]                  Description
        ------------------------------------------------------------------------------------------
        fpath   : string [-]                    Path to the data file.
        lmax : int [days]                       Maximum lag considered.
        lmin : int, optional [days]             Minimum lag considered [default: 0].
        window_size: float, optional [int]      Sliding window size with which to break time series [default: 1].
        window_step: float, optional [int]      Sliding window step [default: 1].
        dsize: int, optional [-]                Down-sampling factor [default: 4].
        ==========================================================================================
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    print('Preparing directory...')

    # obtain data from fpath
    info = os.path.basename(fpath).split("_")
    fld, tstart, tend = info[0], info[1], info[2]
    opath = os.path.dirname(fpath) + f'/CM_{fld}_w{window_size}_s{window_step}_l{lmin}to{lmax}_{info[1]}_{info[2]}.h5'

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

        # latitude and longitude
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        # reduced latitude and longitude
        rlat = ski.measure.block_reduce(lat, block_size=dsize, func=np.mean)
        rlon = ski.measure.block_reduce(lon, block_size=dsize, func=np.mean)
        # delete unnecessary arrays
        del lat, lon

        # time
        t = f['time'][:]
        # time-step
        dt = t[1] - t[0]

        # number of points
        nlat, nlon, nt = len(rlat), len(rlon), len(t)

        # other
        isize = int(window_size / dt)  # iterations/slice
        istep = int(window_step / dt)  # iterations step
        if istep != 0:
            nsteps = int((nt - isize)/istep + 1)  # n steps
        else:
            nsteps = 1
        lmax = int(lmax/dt)  # max lag in iter
        lmin = int(lmin/dt)  # min lag in iter

        print(f"* time segments cover: {(nsteps-1)*istep*dt+isize*dt} days from {tstart} to {tend}")

        with h5py.File(opath, mode='a') as store:
            store.create_dataset("latitude", data=rlat)
            store.create_dataset("longitude", data=rlon)

        # Calculating correlation matrix ...............................................................................

        print("Calculating correlation matrix...")

        for i in range(nsteps):
            tstart = t[i * istep]
            tend = t[i * istep + isize - 1]
            print(f"* segment ({tstart}-{tend} days): {i+1}/{nsteps}")

            # find matrix
            data = f["data"][:, :, i*istep:i*istep+isize-1]

            # reduce matrix
            rdata = ski.measure.block_reduce(data, block_size=(dsize, dsize, 1), func=np.mean)

            # squeeze matrix
            rsdata = rdata.reshape(-1, isize-1)

            # standarise
            ddata = (rsdata - np.mean(rsdata, axis=1).reshape(-1, 1))/np.std(rsdata, axis=1).reshape(-1, 1)

            del data, rdata, rsdata

            # calculate adjacency matrix
            mcorr, mlag = PCC(ddata, max_lag=lmax, min_lag=lmin)
            mlag = mlag * dt  # from steps to hours

            # save adjacency matrix
            key = "t_" + str(tstart) + "_" + str(tend) + "_" + str(i)
            with h5py.File(opath, mode='a') as store:
                store.create_dataset(key, data=mcorr)
                store.create_dataset(key+"_lags", data=mlag)

#%%
ss = [600]
times = {100: (1700, 2000), 200: (1000, 1300), 400: (1200, 1500), 600: (1600, 1900), 800: (1150, 1450), 1000: (1450, 1750), 1200: (1700, 2000)}
for i, s in enumerate(ss):
    print(f"* {s}:")
    main(f"/Volumes/Data/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/q_{1000}_{2000}", lmax=0,
         window_size=1000, window_step=00, dsize=2)
