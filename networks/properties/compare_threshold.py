import h5py
import numpy as np
import pandas as pd
from network_properties import *
from plotting import *

def main(filename, output, indep_var, indep_var_name, dep_var_name):
    with h5py.File(filename, mode='r') as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
        df = pd.DataFrame(columns=[indep_var_name, dep_var_name])
        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"latitude", "longitude"} - keys_lags
        for val in indep_var:
            print(f"{val}:")
            for k in keys_data:
                # load
                cm = f[k][:]  # get correlation data
                np.fill_diagonal(cm, 0)  # take out diagonal
                cm[np.abs(cm) <= val] = 0  # impose threshold
                cm = np.where(cm > 0, 1, np.where(cm < 0, -1, 0))  # unweighted matrix
                # calculate
                values = calc_distances(cm, lat, lon)  # calculate density
                # append
                new_rows = pd.DataFrame({indep_var_name: [val]*len(values), dep_var_name: values})
                df = new_rows.copy() if df.empty else pd.concat([df, new_rows], ignore_index=True)
        #plot
        print("Creating line plot...")
        #plot_line(df, output+"line.png")
        print("Creating histogram plot...")
        plot_hist(df, output+"hist.png")
        print("Creating distance plot...")
        plot_dist(df, output+"dist.png")

ss = [100, 200, 400, 600, 800, 1000, 1200]
seg = 1
l = 7
s = 100
main(f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/CM_q_s{seg}_l0to{l}_1000_2000.h5",
     f"../../../dataloc/pv50-nu4-urlx.c0sat{s}.T170/netdata/",
     [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], "threshold", "distance (km)")
