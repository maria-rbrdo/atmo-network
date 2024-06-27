import h5py
import numpy as np
import pandas as pd
from network_properties import *
from plotting import *

def main(filename, output, indep_var, indep_var_name, dep_var_name):
    with h5py.File(filename, mode='r') as f:
        theta = f["theta"][:]
        phi = f["phi"][:]
        lon = theta  # longitude in rad
        lat = (np.pi / 2 - phi)  # latitude in rad
        df = pd.DataFrame(columns=[indep_var_name, dep_var_name])
        keys_lags = {k for k in f.keys() if k.endswith("_lags")}
        keys_data = set(f.keys()) - {"theta", "phi"} - keys_lags
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

main("../../data/euler/SWE_corr/n1e5_u80_h120_m64/CM_SWE_velocity_PCC_s1_l24.h5",
     "../../data/euler/SWE_corr/n1e5_u80_h120_m64/",
     [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], "threshold", "distance (km)")
