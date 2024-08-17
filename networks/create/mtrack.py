import os
import h5py
import numpy as np
from alive_progress import alive_bar

def iposition(ylim, xlim, npart):
    """
    Set initial position particles.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type      Description
    ------------------------------------------------------------------------------------------
    latlim  : list      Latitude limits.
    lonlim  : list      Longitude limits.
    npart   : int       Number of particles.
    ==========================================================================================
    """
    xi = np.random.rand(npart, 2)
    xi[:, 0] = xi[:, 0] * (max(ylim) - min(ylim)) + min(ylim)
    xi[:, 1] = xi[:, 1] * (max(xlim) - min(xlim)) + min(xlim)
    return xi

def advect(y, x, u, v, dt, sph, ylim=None, xlim=None):
    """
    Advect particles from (lon, lat) to (nx, lat) according to the field (u, v) with
    timestep dt

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name        : Type              Description
    ------------------------------------------------------------------------------------------
    x           : numpy.ndarray     Horizontal coord of particles position.
    y           : numpy.ndarray     Vertical coord of particles position.
    u           : numpy.ndarray     Horizontal velocity at particles position.
    v           : numpy.ndarray     Vertical velocity at particles position.
    dt          : float             Time step.
    sph         : bool              Whether to use spherical coord.
    xlim, ylim  : numpy.ndarray     Cartesian limits.
    ==========================================================================================
    """

    R = 6.371e6  # radius Earth m

    if sph:
        du = u * 24 * 3600 * dt  # distance covered eastward
        dv = v * 24 * 3600 * dt  # distance northward

        ny = y + dv / (2 * np.pi * R) * 360  # new latitude
        nx = x + du / (2 * np.pi * R) * 360  # new longitude
    else:
        ny = y + v * dt  # distance northward
        nx = x + u * dt  # distance covered eastward

    if sph:
        # fix lon out of range
        nx[np.where(nx > 360)] = nx[np.where(nx > 360)] - np.floor(nx[np.where(nx > 360)]/360) * 360
        nx[np.where(nx < 0)] = np.abs(np.floor(nx[np.where(nx < 0)]/360)) * 360 + nx[np.where(nx < 0)]

        # fix lat out of range (assuming not more than one loop)
        nx[np.where(ny > 90)] = ((nx[np.where(ny > 90)] + 180) * (nx[np.where(ny > 90)] < 180)
                                 + (nx[np.where(ny > 90)] - 180) * (nx[np.where(ny > 90)] >= 180))
        ny[np.where(ny > 90)] = 2 * 90 - ny[np.where(ny > 90)]

        nx[np.where(ny < -90)] = ((nx[np.where(ny < -90)] + 180) * (nx[np.where(ny < -90)] < 180)
                                  + (nx[np.where(ny < -90)] - 180) * (nx[np.where(ny < -90)] >= 180))
        ny[np.where(ny < -90)] = -2 * 90 - ny[np.where(ny < -90)]

    else:
        nx[np.where(nx > xlim[1])] = (nx[np.where(nx > xlim[1])]
                                      - np.floor(nx[np.where(nx > xlim[1])]/(xlim[1] - xlim[0])) * (xlim[1] - xlim[0]))
        nx[np.where(nx < xlim[0])] = (np.abs(np.floor(nx[np.where(nx < xlim[0])]/(xlim[1] - xlim[0])) * (xlim[1] - xlim[0]))
                                      + nx[np.where(nx < xlim[0])])

        ny[np.where(ny > ylim[1])] = (ny[np.where(ny > ylim[1])]
                                      - np.floor(ny[np.where(ny > ylim[1])]/(ylim[1] - ylim[0])) * (ylim[1] - ylim[0]))
        ny[np.where(ny < ylim[0])] = (np.abs(np.floor(ny[np.where(ny < ylim[0])]/(ylim[1] - ylim[0])) * (ylim[1] - ylim[0]))
                                      + ny[np.where(ny < ylim[0])])
    return ny, nx

def track(upath, vpath, npart, nt, dt, sph, ylim=(0, 90), xlim=(0, 360)):
    """
    Track npart advected by a velocity field.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type          Description
    ------------------------------------------------------------------------------------------
    upath   : string        Path to eastward velocity field file.
    vpath   : string        Path to northward velocity field file.
    npart   : int           Number of particles.
    nt      : int           Number of timesteps.
    dt      : float         Time step.
    sph     : bool          Whether to use spherical coord.
    ==========================================================================================
    """

    with h5py.File(upath, mode='r') as fu, h5py.File(vpath, mode='r') as fv:

        if sph:
            y = fu['latitude'][:]
            x = fu['longitude'][:]
        else:
            y = fu['y'][:]
            x = fu['x'][:]

        X = np.empty((npart, 2, nt))
        X[:, :, 0] = iposition(ylim, xlim, npart)

        print("Calculating particles trajectories:")
        with alive_bar(nt-1, force_tty=True) as bar:
            for i in range(nt - 1):
                yidx = [np.nanargmin((y - X[n, 0, i]) ** 2) for n in range(npart)]
                xidx = [np.nanargmin((x - X[n, 1, i]) ** 2) for n in range(npart)]

                u = np.array([fu['data'][yidx[n], xidx[n], i] for n in range(npart)])
                v = np.array([fv['data'][yidx[n], xidx[n], i] for n in range(npart)])

                X[:, 0, i + 1], X[:, 1, i + 1] = advect(X[:, 0, i], X[:, 1, i], u, v, dt, sph=sph, ylim=ylim, xlim=xlim)
                bar()

        return X

def calc_distance_sph(lat1, lat2, lon1, lon2):
    """
    Calculate distance between two points.
    """
    lat1, lat2, lon1, lon2 = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon1), np.deg2rad(lon2)
    R = 6.371e6  # m radius Earth
    Dlat = np.abs(lat1 - lat2)
    Dlon = np.abs(lon1 - lon2)
    a = np.sin(Dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(Dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-np.round(a, 15)))
    return R*c

def calc_distance_cart(y1, y2, x1, x2):
    """
    Calculate distance between two points.
    """
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def madj(coord, sph=True):
    """
    Calculate adjacency matrix.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name   : Type              Description
    ------------------------------------------------------------------------------------------
    coord  : numpy.ndarray     Array containing coordinates particle paths (npart, 2, nsteps).
    sph    : bool              Whether to use spherical coord.
    ==========================================================================================
    """

    adj = np.empty((coord.shape[0], coord.shape[0]))

    print("Calculating adjacency matrix:")
    with alive_bar(coord.shape[0], force_tty=True) as bar:
        for i, a in enumerate(coord):
            for j, b in enumerate(coord[i:], start=i):
                # calculate distance between trajectories
                if sph:
                    dist = calc_distance_sph(a[0, :], b[0, :], a[1, :], b[1, :])
                else:
                    dist = calc_distance_cart(a[0, :], b[0, :], a[1, :], b[1, :])
                # calculate mean distance
                avdist = np.mean(dist)
                # calculate adjacency matrix entry
                if avdist != 0:
                    adj[i, j] = (1/avdist) * np.sum((avdist - dist) ** 2) ** (1/2)
                else:
                    adj[i, j] = 0
                # adjacency matrix symmetry
                if i != j:
                    adj[j, i] = adj[i, j]
            bar()
    return adj

def main(upath, vpath, npart, sph, ylim=[0,90], xlim=[0,360]):

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    print('Preparing directory...')

    # obtain data from fpath
    if sph:
        info1 = os.path.basename(upath).split("_")
        info2 = os.path.basename(vpath).split("_")
        assert info1[1] == info2[1] and info1[2] == info2[2], "files do not correspond to the same time segment"
        opath = os.path.dirname(upath) + f'/TM_{info1[1]}_{info1[2]}.h5'
    else:
        opath = os.path.dirname(upath) + f'/TM.h5'
    # delete files from previous runs
    try:
        os.remove(opath)
        print(f"Previous file '{opath}' deleted successfully.")
    except:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Load data:
    # ------------------------------------------------------------------------------------------------------------------
    with h5py.File(upath, mode='r') as f:
        if sph:
            y = f['latitude'][:]
            x = f['longitude'][:]
        else:
            y = f['y'][:]
            x = f['x'][:]
        t = f['time'][:]
        dt = t[1] - t[0]
        nt = len(t)

        with h5py.File(opath, mode='a') as store:
            if sph:
                store.create_dataset("latitude", data=y)
                store.create_dataset("longitude", data=x)
            else:
                store.create_dataset("y", data=y)
                store.create_dataset("x", data=x)

        del y, x

    coord = track(upath, vpath, npart, nt, dt, sph=sph, ylim=ylim, xlim=xlim)
    mtrack = madj(coord, sph=sph)

    with h5py.File(opath, mode='a') as store:
        store.create_dataset("coord", data=coord)
        store.create_dataset("adj", data=mtrack)


main("/Volumes/Data/dataloc/pv50-nu4-urlx.c0sat200.T170_highres/netdata/u_1140_1160",
     "/Volumes/Data/dataloc/pv50-nu4-urlx.c0sat200.T170_highres/netdata/v_1140_1160",
     2000, ylim=[15, 90], xlim=[0, 360], sph=True)

#main("/Volumes/Maria/dataloc/quadgyre/netdata/u_e0_s500_t15",
#     "/Volumes/Maria/dataloc/quadgyre/netdata/v_e0_s500_t15",
#     1000, ylim=[-1, 1], xlim=[0, 2], sph=False)



