import h5py
import numpy as np
import pandas as pd
from network_properties import *
from plotting import *

def main(foldername, output, indep_var, indep_var_name, dep_var_name):
    df = pd.DataFrame(columns=[indep_var_name, dep_var_name])
    for val in indep_var:
        print(f"{val}:")
        filename = foldername + f"n1e5_u{val}_h120_m64/CM_SWE_vorticity_PCC_s1_l24.h5"
        with h5py.File(filename, mode='r') as f:
            theta = f["theta"][:]
            phi = f["phi"][:]
            lon = theta  # longitude in rad
            lat = (np.pi / 2 - phi)  # latitude in rad

            for k in set(f.keys()) - {"theta", "phi"}:
                # load
                cm = f[k][:]  # get correlation data
                np.fill_diagonal(cm, 0)  # take out diagonal
                cm[np.abs(cm) <= 0.9] = 0  # impose threshold
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

main("../../data/euler/SWE_corr/",
     "../../data/euler/SWE_corr/",
     [20, 30, 40, 50, 60, 70], "u_max", "distance (km)")
