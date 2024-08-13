"""
========================================================================================================================
Network Evolution Script
========================================================================================================================
This script calculates graph measures throughout time.
------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
from alive_progress import alive_bar

from networks.calculate.netprop import *
from networks.plot.plot import *

def main(fname, opath, measure, tau=0):
    """
        This function runs the script.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name        : Type                  Description
        ------------------------------------------------------------------------------------------
        fpath       : string                Path to the data file.
        opath       : string                Path to the output directory.
        measure     : string                Network measure to plot.
        tau         : int, optional         Threshold to apply [default: 0].
        ==========================================================================================
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare directory:
    # ------------------------------------------------------------------------------------------------------------------

    oname = f"{fname.split("/")[-1].split(".")[0]}"
    folder = os.path.join(opath + oname +"_evol/")
    ofile = f"{folder}{oname}_evol_t{tau}.h5"
    if not os.path.exists(folder):
        os.mkdir(folder)

    # ------------------------------------------------------------------------------------------------------------------
    # Run:
    # ------------------------------------------------------------------------------------------------------------------

    # Initialise arrays ................................................................................................

    vals = []
    time = []

    with h5py.File(fname, mode='r') as fdata:

        # Load data ....................................................................................................

        lat = fdata["latitude"][:]
        lon = fdata["longitude"][:]
        nlat, nlon = len(lat), len(lon)
        del lat, lon

        keys_lags = {k for k in fdata.keys() if k.endswith("_lags")}
        keys_data = set(fdata.keys()) - {"longitude", "latitude"} - keys_lags

        # Iterate ......................................................................................................
        with alive_bar(int(len(keys_data)), force_tty=True) as bar:
            for k in sorted(keys_data):

                # Load data ............................................................................................
                am = fdata[k][:]  # get data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                times = [int(s) for s in k.split('_') if s.isdigit()]  # get times
                if len(times) > 2:
                    time.append((times[0]+times[1])/2)
                else:
                    time.append(times[0])

                # Measure and store ....................................................................................

                # Strength
                if measure == "strength" or measure == "in_strength":
                    net, _ = calc_strength(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "out_strength":
                    _, net = calc_strength(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "eigenvector":
                    net = calc_eigenvector(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "closeness":
                    net = calc_closeness(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "betweenness":
                    net = calc_betweenness(am, nlat, nlon)
                    vals.append(net.reshape(-1))
                elif measure == "clustering":
                    val = calc_clustering(am)
                    vals.append(val)
                elif measure == "density":
                    val = calc_density(am)
                    vals.append(val)
                else:
                    raise "Not recognised network measure."

                # %% Update bar
                bar()

    with h5py.File(ofile, mode='a') as store:
        try:
            del store[measure], store[measure+"_t"]
        except:
            pass
        store.create_dataset(measure, data=vals)
        store.create_dataset(measure+"_t", data=time)

#if __name__ == "__main__":

#    args = docopt(__doc__)

#    main(model=args['<model>'], task=args['<task>'], method=args['<method>'], measure=args['<measure>'],
#         lag=args['--lag'], tau=float(args['--tau']), degree_distribution=bool(args['--degree_distribution'] == "True"),
#         filename=args['<files>'], output=args['--output'])



type = "window_zero"  # CHECK
measures = ["density", "clustering", "strength", "closeness", "betweenness"]
s = 600

ss = [100, 200, 400, 600, 800, 1000, 1200]
times = {100: (1700, 2000), 200: (1700, 2000), 400: (1200, 1500), 600: (1600, 1900), 800: (1150, 1450), 1000: (1450, 1750), 1200: (1700, 2000)}
if type == "all_zero":  # rho = 0.1
    thresh = {100: 0.455, 200: 0.837, 400: 0.324, 600: 0.373, 800: 0.331, 1000: 0.599, 1200: 0.109}
elif type == "window_zero":  # rho = 0.05
    thresh = {100: 0.695, 200: 0.751, 400: 0.601, 600: 0.526, 800: 0.542, 1000: 0.597, 1200: 0.508}
elif type == "vort":
    thresh = {100: 0, 200: 0, 400: 0, 600: 0, 800: 0, 1000: 0, 1200: 0}

for s in ss:
    print(f"* {s}:")
    if s == 100:
        print(f"** betweenness:")
        main(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_w25_s10_l0to0_{times[s][0]}_{times[s][1]}.h5",
             f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/", "betweenness", tau=thresh[s])
    else:
        for m in measures:
            print(f"** {m}:")
            main(f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/CM_q_w25_s10_l0to0_{times[s][0]}_{times[s][1]}.h5",
                 f"/Volumes/Results/dataloc/pv50-nu4-urlx.c0sat{s}.T170/", m, tau=thresh[s])
#for i, s in enumerate(ss):
#    main(f"/home/reboredoprad/bob/dataloc/bb/swvac/pv50-nu4-urlx.c0sat{s}.T170/netdata/q_{times[i][0]}_{times[i][1]}",
#         f"/home/reboredoprad/bob/dataloc/bb/swvac/pv50-nu4-urlx.c0sat{s}.T170/netdata/", "betweenness",
#         tau=thresh[s])