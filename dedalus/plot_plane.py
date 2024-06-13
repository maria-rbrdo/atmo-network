"""
ENVIRONMENT: dedalus3

This script plots the results of the dedalus SWE/TSWE simulations on a plane.
Plots are saved as png files in the output directory specified.

To plot the data using e.g. 1 processes:
    $ mpiexec -n 1 python3 plot_plane.py data/model/SWE_snapshots/*.h5
    $ mpiexec -n 1 python3 plot_plane.py data/model/TSWE_snapshots/*.h5

To make a movie with resulting files run:
    $ import os
    $ output = "data/model/SWE_frames/n1e5_u10_h120_m64"
    $ os.system(f"ffmpeg -r 1 -i {output}/write_%06d.png -vcodec mpeg4 -y {output}/movie.mp4")

Usage:
    dedalus/plot_plane.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./data/model/frames_plane]

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib
import seaborn as sns
from alive_progress import alive_bar


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    """Formatter for showing the colorbar in scientific computing"""
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format

def main(filename, start, count, output, movie = True):
    """Create plane plots of height and vorticity"""

    # select tasks
    tasks = ['height', 'temperature', 'vorticity']
    tasks = ['height', 'vorticity']
    n_tasks = len(tasks)

    # plot specifics
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f} hrs'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # create figure
    fig = plt.figure(figsize=(20, 5*n_tasks+1))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 25})

    # subplots
    gs0 = matplotlib.gridspec.GridSpec(n_tasks, 1, figure=fig)

    # plot writes
    with h5py.File(filename, mode='r') as file:
        with alive_bar(manual=True, force_tty=True) as bar:
            for index in range(start, start+count):
                for n, task in enumerate(tasks):

                    # axis position
                    ax = fig.add_subplot(gs0[n])
                    if n == 0:
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0 + 0.025, box.width, box.height])
                    elif n == n_tasks-1:
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0 - 0.025, box.width, box.height])

                    # plot
                    dset = file['tasks'][task]
                    theta = dset.dims[1][0][:]
                    phi = dset.dims[2][0][:]
                    lat = (np.pi / 2 - phi)*180/np.pi  # define latitude
                    lon = (np.pi - theta)*180/np.pi  # define longitude

                    data = dset[:]  # get data
                    base10 = np.floor(np.log10(abs(np.max(data))))  # power of ten data
                    sns.heatmap(np.flip(data[index, :, :].T, 0), cmap="vlag", ax=ax, center=0,
                                cbar_kws=dict(use_gridspec=False, location="top", aspect=60, extend='both',
                                              format=OOMFormatter(base10, mathText=False), label=task, pad=0.01))

                    # specify ticks plot
                    x_ticks = 9
                    y_ticks = 5
                    ax.set_xticks(np.linspace(0, len(lon), x_ticks))
                    ax.set_xticklabels(np.linspace(-180, 180, x_ticks, dtype=int))
                    ax.set_yticks(np.linspace(0, len(lat), y_ticks))
                    ax.set_yticklabels(np.linspace(90, -90, y_ticks, dtype=int))
                    ax.set_xlabel('longitude (deg)')
                    ax.set_ylabel('latitude (deg)')

                # add time title
                fig.suptitle(title_func(file['scales/sim_time'][index]))

                # save figure
                savename = savename_func(file['scales/write_number'][index])
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi, bbox_inches='tight')
                fig.clear()

                # update bar
                bar((index-start)/(count-1))
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
