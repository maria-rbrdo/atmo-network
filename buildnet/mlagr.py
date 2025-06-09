"""
====================================================================================================
Lagrangian Adjacency Matrix Building Script
====================================================================================================
This script reads the HDF5 net-output file from the model and finds the lagrangian-based adjacency
matrices. Data is saved to a HDF5 file in the output directory specified.
------------------------------------------------------------------------------------------------------------------------
Notes:
- This code is written to for a spherical domain. If the domain is not spherical this network is not well-built.
------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator

# -------------------------------------------------------------------------------------------------
# Functions:
# -------------------------------------------------------------------------------------------------

def iposition(ylim, xlim, npart):
    """
    Set initial position particles.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name    : Type      Description
    -----------------------------------------------------------------------------------------------
    latlim  : list      Latitude limits.
    lonlim  : list      Longitude limits.
    npart   : int       Number of particles.
    ===============================================================================================
    """
    xi = np.random.rand(npart, 2)
    xi[:, 0] = xi[:, 0] * (max(ylim) - min(ylim)) + min(ylim)
    xi[:, 1] = xi[:, 1] * (max(xlim) - min(xlim)) + min(xlim)
    return xi

def vinterpolate(yy, xx, u, qx):
    """
    Interpolate velocity.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name        : Type              Description
    -----------------------------------------------------------------------------------------------
    xx          : numpy.ndarray     Longitudes on the grid.
    yy          : numpy.ndarray     Latitudes on the grid.
    u           : numpy.ndarray     Velocity field.
    qx          : numpy.ndarray     Query points.
    ===============================================================================================
    """
    vinterp = LinearNDInterpolator(list(zip(xx.ravel(), yy.ravel())), u[0, :, :].ravel())
    uinterp = LinearNDInterpolator(list(zip(xx.ravel(), yy.ravel())), u[1, :, :].ravel())

    V = vinterp(qx[:, 1], qx[:, 0])
    U = uinterp(qx[:, 1], qx[:, 0])
    return np.array([V, U]).T

def qinterpolate(yy, xx, pv, qx):
    """
    Interpolate PV.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name        : Type              Description
    -----------------------------------------------------------------------------------------------
    xx          : numpy.ndarray     Longitudes on the grid.
    yy          : numpy.ndarray     Latitudes on the grid.
    pv          : numpy.ndarray     PV field.
    qx          : numpy.ndarray     Query points.
    ===============================================================================================
    """

    #interpolate
    qinterp = LinearNDInterpolator(list(zip(xx.ravel(), yy.ravel())), pv.ravel())
    PV = qinterp(qx[:, 1], qx[:, 0])

    # deal with trajectories close to the poles
    if np.any(np.isnan(PV)):
        idx = np.unique(np.where(np.isnan(PV))[0])
        idx_out = idx[np.where(qx[idx, 0] > 89)]
        idx_in = list(set(idx) - set(idx_out))
        if idx_in:
            raise ValueError
        else:
            PV[idx_out] = np.mean(pv[yy == np.max(yy)])

    return PV
def rk4(yy, xx, un, unp1, xn, dt, deg=False):
    """
    Take a 4th order Runge-Kutta step from xn to xnp1 according to the velocity field u with
    timestep dt.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name        : Type              Description
    -----------------------------------------------------------------------------------------------
    xx          : numpy.ndarray     Longitudes on the grid.
    yy          : numpy.ndarray     Latitudes on the grid.
    un          : numpy.ndarray     Velocities at timestep n.
    unp1        : numpy.ndarray     Velocities at timestep n+1.
    xn          : numpy.ndarray     Position of the particles.
    dt          : float             Time step.
    ===============================================================================================
    """

    # RK4 for dx/dt = u dy/dt = v
    if deg:
        # k1 -> f(tn, xn)
        k1 = vinterpolate(yy, xx, un, xn % 360)
        # k2 -> f(tn + dt/2, xn + dt*k1/2)
        k2 = vinterpolate(yy, xx, (un+unp1)/2, (xn+dt*k1/2) % 360)
        # k3 -> f(tn + dt/2, xn + k2/2)
        k3 = vinterpolate(yy, xx, (un+unp1)/2, (xn+dt*k2/2) % 360)
        # k4 -> f(tn + dt, xn + k3)
        k4 = vinterpolate(yy, xx, unp1, (xn+dt*k3) % 360)
        # step:
        xnp1 = (xn + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6) % 360
    else:
        # k1 -> f(tn, xn)
        k1 = vinterpolate(yy, xx, un, xn)
        # k2 -> f(tn + dt/2, xn + dt*k1/2)
        k2 = vinterpolate(yy, xx, (un+unp1)/2, xn+dt*k1/2)
        # k3 -> f(tn + dt/2, xn + k2/2)
        k3 = vinterpolate(yy, xx, (un+unp1)/2, xn+dt*k2/2)
        # k4 -> f(tn + dt, xn + k3)
        k4 = vinterpolate(yy, xx, unp1, xn+dt*k3)
        # step:
        xnp1 = xn + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return xnp1

def advect(yy, xx, Xn, un, unp1, dt):
    """
    Advect particles from Xn to Xnp1 according to the velocity field u with timestep dt.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name        : Type              Description
    -----------------------------------------------------------------------------------------------
    xx          : numpy.ndarray     Longitudes on the grid.
    yy          : numpy.ndarray     Latitudes on the grid.
    Xn          : numpy.ndarray     Position of particles.
    un          : numpy.ndarray     Velocities at timestep n.
    unp1        : numpy.ndarray     Velocities at timestep n+1.
    dt          : float             Time step.
    ===============================================================================================
    """

    R = 6.371e6  # radius Earth m

    unr = un / R * 180/np.pi
    unr[1, :, :] = unr[1, :, :] / np.cos(np.deg2rad(yy))
    unp1r = unp1 / R * 180/np.pi
    unp1r[1, :, :] = unp1r[1, :, :] / np.cos(np.deg2rad(yy))

    Xnp1 = rk4(yy, xx, unr, unp1r, Xn, dt, deg=True)
    Xnp1[np.isnan(Xnp1)] = Xn[np.isnan(Xnp1)]

    # nodes close to poles ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    Rp = 15  # radius in deg
    nhp = (Xn[:, 0] > 90-Rp) | (Xnp1[:, 0] > 90-Rp)  # points near nh

    if nhp.size > 0:

        yy, xx = yy[:-1, :], xx[:-1, :]
        un, unp1 = un[:, :-1, :], unp1[:, :-1, :]

        # coord on polar stereographic plane
        Xnr, xxr, yyr = np.deg2rad(Xn), np.deg2rad(xx), np.deg2rad(yy)

        # create arrays
        Xn_nhp = np.zeros_like(Xn[nhp, :])
        un_nhp = np.zeros_like(un)
        unp1_nhp = np.zeros_like(unp1)

        # to plane coordinates
        xx_nhp = - R * np.cos(xxr) / np.tan(yyr)
        yy_nhp = - R * np.sin(xxr) / np.tan(yyr)
        Xn_nhp[:, 1] = - R * np.cos(Xnr[nhp, 1]) / np.tan(Xnr[nhp, 0])
        Xn_nhp[:, 0] = - R * np.sin(Xnr[nhp, 1]) / np.tan(Xnr[nhp, 0])

        # vel on polar stereographic plane
        un_nhp[1, :, :] = - np.sin(xxr) / np.sin(yyr) * un[1, :, :] - np.cos(xxr) / (np.sin(yyr) ** 2) * un[0, :, :]
        un_nhp[0, :, :] = + np.cos(xxr) / np.sin(yyr) * un[1, :, :] - np.sin(xxr) / (np.sin(yyr) ** 2) * un[0, :, :]
        unp1_nhp[1, :, :] = - np.sin(xxr) / np.sin(yyr) * unp1[1, :, :] - np.cos(xxr) / (np.sin(yyr) ** 2) * unp1[0, :, :]
        unp1_nhp[0, :, :] = + np.cos(xxr) / np.sin(yyr) * unp1[1, :, :] - np.sin(xxr) / (np.sin(yyr) ** 2) * unp1[0, :, :]

        # new coord
        Xnp1_nhp = rk4(yy_nhp, xx_nhp, -un_nhp, -unp1_nhp, Xn_nhp, dt)

        # transform back
        Xnp1[nhp, 1] = np.rad2deg(np.arctan2(Xnp1_nhp[:, 0], Xnp1_nhp[:, 1])) + 180
        Xnp1[nhp, 0] = - np.rad2deg(- np.arctan(R / np.sqrt(Xnp1_nhp[:, 1]**2 + Xnp1_nhp[:, 0]**2)))

    return Xnp1

def track(upath, vpath, qpath, npart, nt, dt, ylim=(0, 90), xlim=(0, 360)):
    """
    Track npart advected by a velocity field for nt timesteps of size dt.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name    : Type          Description
    -----------------------------------------------------------------------------------------------
    upath   : string        Path to velocity field file.
    vpath   : string        Path to velocity field file.
    qpath   : string        Path to potential vorticity field file.
    npart   : int           Number of particles.
    nt      : int           Number of timesteps.
    dt      : float         Time step.
    ===============================================================================================
    """

    with (h5py.File(upath, mode='r') as fu, h5py.File(vpath, mode='r') as fv, h5py.File(qpath, mode='r') as fq):

        # Initialise ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        y, x = fu['latitude'][:], fu['longitude'][:]

        px = np.concatenate((np.atleast_1d(0), x, np.atleast_1d(360)))
        xx, yy = np.meshgrid(px, y)
        del px

        X = np.zeros((npart, 2, nt))
        X[:, :, 0] = iposition(ylim, xlim, npart)

        PV = np.zeros((npart, nt))
        q0 = np.concatenate((np.atleast_2d(fq['data'][:, 0, 0]).T, fq['data'][:, :, 0],
                             np.atleast_2d(fq['data'][:, -1, 0]).T), axis=1)
        PV[:, 0] = qinterpolate(yy, xx, q0, X[:, :, 0])

        # Loop ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        for i in tqdm(range(nt-1), desc='Calculating trajectories...'):
            # Get velocities
            un = np.array([fv['data'][:, :, i], fu['data'][:, :, i]])
            unp1 = np.array([fv['data'][:, :, i+1], fu['data'][:, :, i+1]])

            # Complete range
            un = np.concatenate((np.atleast_3d(un[:, :, 0]), un, np.atleast_3d(un[:, :, -1])), axis=2)
            unp1 = np.concatenate((np.atleast_3d(unp1[:, :, 0]), unp1, np.atleast_3d(unp1[:, :, -1])), axis=2)

            # Advect
            X[:, :, i + 1] = advect(yy, xx, X[:, :, i], un, unp1, dt * 24 * 3600)

            # Store PV
            qnp1 = np.concatenate((np.atleast_2d(fq['data'][:, 0, i]).T, fq['data'][:, :, i],
                                   np.atleast_2d(fq['data'][:, -1, i]).T), axis=1)
            PV[:, i+1] = qinterpolate(yy, xx, qnp1, X[:, :, i+1])

            # Remove trajectories going onto the equator/SH
            didx = np.argwhere(X[:, 0, i+1] < 1)
            X = np.delete(X, didx, axis=0)
            PV = np.delete(PV, didx, axis=0)

        return X, PV

def calc_distance_sph(lat1, lat2, lon1, lon2):
    """
    Calculate distance between two points.
    """

    lat1, lat2, lon1, lon2 = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)
    R = 6.371e6  # m radius Earth
    Dlat = np.abs(lat1 - lat2)
    Dlon = np.abs(lon1 - lon2)
    a = np.sin(Dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(Dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - np.round(a, 15)))
    return R * c

def mprox(coord):
    """
    Calculate proximity adjacency matrix.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name   : Type              Description
    -----------------------------------------------------------------------------------------------
    coord  : numpy.ndarray     Array containing coordinates particle paths (npart, 2, nsteps).
    ===============================================================================================
    """

    adj = np.empty((coord.shape[0], coord.shape[0]))

    print("Calculating proximity adjacency matrix:")
    for i, a in enumerate(tqdm(coord)):
        for j, b in enumerate(coord[i:], start=i):
            # calculate distance between trajectories
            dist = calc_distance_sph(a[0, :], b[0, :], a[1, :], b[1, :])
            # calculate adjacency matrix entry
            adj[i, j] = np.sum(dist ** 2) ** (1/2)
            # adjacency matrix symmetry
            if i != j:
                adj[j, i] = adj[i, j]
    return adj

def mdynsim(q):
    """
    Calculate dynamic similarity adjacency matrix.
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name   : Type              Description
    -----------------------------------------------------------------------------------------------
    q      : numpy.ndarray     Array containing pv of particle paths (npart, 1, nsteps).
    ===============================================================================================
    """

    adj = np.empty((q.shape[0], q.shape[0]))

    print("Calculating proximity adjacency matrix:")
    for i, a in enumerate(tqdm(q)):
        for j, b in enumerate(q[i:], start=i):
            # calculate adjacency matrix entry
            adj[i, j] = np.mean(np.abs(a - b))
            # adjacency matrix symmetry
            if i != j:
                adj[j, i] = adj[i, j]
    return adj

# -------------------------------------------------------------------------------------------------
# Main:
# -------------------------------------------------------------------------------------------------

def main(upath, vpath, qpath, npart, ylim=[0,90], xlim=[0,360]):
    """
    ===============================================================================================
    Parameters :
    -----------------------------------------------------------------------------------------------
    Name    : Type          Description
    -----------------------------------------------------------------------------------------------
    upath   : string        Path to velocity field file.
    vpath   : string        Path to velocity field file.
    qpath   : string        Path to potential vorticity field file.
    npart   : int           Number of particles.
    ===============================================================================================
    """

    # Prepare directory :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    print('Preparing directory...')

    opath = os.path.dirname(upath) + f'/../matrices/'
    if not os.path.exists(opath):
        os.makedirs(opath)

    # delete files from previous runs
    opath = os.path.dirname(upath) + f'/../matrices/LM.h5'
    flag = True
    i = 1
    while flag:
        if os.path.exists(opath):
            i += 1
            opath = os.path.dirname(upath) + f'/../matrices/LM_{i}.h5'
        else:
            flag = False

    # Load data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    with h5py.File(upath, mode='r') as f:
        y = f['latitude'][:]
        x = f['longitude'][:]

        t = f['time'][:]
        nt = len(t)
        dt = t[1] - t[0]

        with h5py.File(opath, mode='a') as store:
            store.create_dataset("latitude", data=y)
            store.create_dataset("longitude", data=x)

        del y, x

    coord, q = track(upath, vpath, qpath, npart, nt, dt, ylim=ylim, xlim=xlim)
    adj1 = mprox(coord)
    adj2 = mdynsim(q)

    with h5py.File(opath, mode='a') as store:
        store.create_dataset("coord", data=coord)
        store.create_dataset("proximity", data=adj1)
        store.create_dataset("dynamic-similarity", data=adj2)

# -------------------------------------------------------------------------------------------------
# Run:
# -------------------------------------------------------------------------------------------------

main(
    upath="~/u",
    vpath="~/v",
    qpath="~/q",
    npart=2000,
    ylim=[15, 90],
    xlim=[0, 360]
)