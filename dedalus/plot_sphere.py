"""
ENVIRONMENT: dedalus3

This script plots the results of the dedalus SWE/TSWE simulations on a sphere.
Plots are saved as png files in the output directory specified.

To plot the data using e.g. 1 processes:
    $ mpiexec -n 1 python3 plot_sphere.py data/model/SWE_snapshots/*.h5
    $ mpiexec -n 1 python3 plot_sphere.py data/model/TSWE_snapshots/*.h5

Usage:
    plot_sphere.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/model/frames_sphere]

"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def build_s2_coord_vertices(theta, phi):
    theta = theta.ravel()
    theta_vert = np.concatenate([theta, [2*np.pi]])
    theta_vert -= theta_vert[1] / 2
    phi = phi.ravel()
    phi_mid = (phi[:-1] + phi[1:]) / 2
    phi_vert = np.concatenate([[np.pi], phi_mid, [0]])
    return np.meshgrid(theta_vert, phi_vert, indexing='ij')


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = 'vorticity'
    cmap = plt.cm.RdBu
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        theta = dset.dims[1][0][:].ravel()
        phi = dset.dims[2][0][:].ravel()
        theta_vert, phi_vert = build_s2_coord_vertices(theta, phi)
        x = np.sin(phi_vert) * np.cos(theta_vert)
        y = np.sin(phi_vert) * np.sin(theta_vert)
        z = np.cos(phi_vert)
        for index in range(start, start+count):
            data = dset[index, :, :]
            clim = np.max(np.abs(data))
            norm = matplotlib.colors.Normalize(-clim, clim)
            fc = cmap(norm(data))
            # fc[:, phi.size//2, :] = [0,0,0,1]  # black equator
            if index == start:
                surf = ax.plot_surface(x, y, z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=5)
                ax.set_box_aspect((1, 1, 1))
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(-0.7, 0.7)
                ax.set_zlim(-0.7, 0.7)
                ax.axis('off')
            else:
                surf.set_facecolors(fc.reshape(fc.size//4, 4))
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
