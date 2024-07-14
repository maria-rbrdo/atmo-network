"""
========================================================================================================================
Threshold Comparison Script
========================================================================================================================
This script builds compares the density / link distance obtained for different thresholds.
------------------------------------------------------------------------------------------------------------------------
"""

import h5py
import numpy as np
import pandas as pd

from netprop import *
from plot import *

def main(filename, output, thresh, dep_var_name):
    with h5py.File(filename, mode='r') as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        glon, glat = np.meshgrid(lon, lat)
        glon, glat = glon.reshape(-1), glat.reshape(-1)

        df = pd.DataFrame(columns=["threshold", dep_var_name])
        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"latitude", "longitude"} - keys_lags
        for tau in thresh:
            print(f"{tau}:")
            for k in keys_data:
                # load
                am = f[k][:]  # get correlation data
                np.fill_diagonal(am, 0)  # take out diagonal
                am = np.abs(am)  # take absolute value
                am[np.abs(am) <= tau] = 0  # impose threshold

                # calculate
                if dep_var_name == "distance (km)":
                    values = calc_distances(am, glat, glon)
                elif dep_var_name == "density":
                    values = calc_density(am)

                # append
                new_rows = pd.DataFrame({"threshold": [tau]*len(values), dep_var_name: values})
                df = new_rows.copy() if df.empty else pd.concat([df, new_rows], ignore_index=True)

        #plot
        print("Creating line plot...")
        plot_line(df, output+"line.png")
        print("Creating histogram plot...")
        #plot_hist(df, output+"hist.png")
        #print("Creating distance plot...")
        #plot_dist(df, output+"dist.png")

ss = [100, 200, 400, 600, 800, 1000, 1200]
seg = 1
l = 7
for s in ss:
    main(f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/CM_q_s{seg}_l0to{l}_1000_2000.h5",
         f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/",
         [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 1], "density")
