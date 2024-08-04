"""
========================================================================================================================
Surrogate Network Script
========================================================================================================================
This script reads the HDF5 net-output file from the model, creates surrogate networks, and finds the correlation-based
adjacency matrix between all grid points for each of them. Then it plots the results.

Correlations can be found at zero lag (lag = False) or at time lag (lag = True). Data is saved to a HDF5 file in the
output directory specified.
------------------------------------------------------------------------------------------------------------------------
"""

import h5py
import numpy as np
import skimage as ski
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
sat = [100, 200, 400, 600, 800, 1000, 1200]
lmax = 0
segments = 1
nsurr = 50
dsize = 4

plot_data = []
a = []
for s in sat:
    with h5py.File(f"/Volumes/Maria/dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/q_1000_2000", mode='r') as f:

        A0 = s / 600 * 0.15
        print(f"* {A0:04.3f}:")

        # Loading data .............................................................................................
        # data
        data = f["data"][:]
        t = f['time'][:]

        # reduced matrices
        rdata = ski.measure.block_reduce(data, block_size=(dsize, dsize, 1), func=np.mean)

        # time-step
        dt = t[1] - t[0]

        # number of points
        nt = len(t)

        # other
        diter = int(nt / segments)  # iterations/slice

        # squeeze matrix
        rsdata = rdata.reshape(-1, diter)

        # delete unnecessary arrays
        del data, rdata, t

        # Calculating correlation matrix ...............................................................................

        # array to store data
        PCC_data = []
        with alive_bar(nsurr, force_tty=True) as bar:
            for n in range(nsurr):

                # Shuffle timeseries
                surr = np.array([rsdata[it, np.random.permutation(rsdata.shape[1])] for it in range(rsdata.shape[0])])

                # standarise
                surr = (surr - np.mean(surr, axis=1).reshape(-1, 1)) / np.std(surr, axis=1).reshape(-1, 1)

                # calculate adjacency matrix
                mcorr, _ = PCC(surr, max_lag=lmax)
                np.fill_diagonal(mcorr, 0)  # take out diagonal
                mcorr = np.abs(mcorr)  # take absolute value

                # save PCC data
                PCC_data.append(mcorr.reshape(-1))
                bar()

        plot_data.append(np.array(PCC_data).reshape(-1))
        a.append(A0)

# PLOT:

plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(figsize=(11, 12))
bplot = ax.boxplot(plot_data, positions=a, showfliers=True, widths=0.02, patch_artist=True, manage_ticks=False,
                   boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2.5), medianprops=dict(color="black"))
ax.plot(a, [np.mean(x) for x in plot_data], "k*", markersize=12)
cmap = cm.ScalarMappable(cmap='rainbow')
for patch, color in zip(bplot['boxes'], cmap.to_rgba(range(len(bplot['boxes'])))):
    patch.set_facecolor(color)
box = ax.get_position()
ax.set_position([box.x0+0.01, box.y0 + box.height * 0.18,
                 box.width, box.height * 0.9])
ax.set_xlabel(f'$A_0/H$')
ax.set_ylabel(f'PCC')
ax.set_xlim([0, 0.32])
ax.set_title("(b)")
plt.savefig("apcc.png", dpi=200)
