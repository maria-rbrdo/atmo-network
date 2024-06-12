import h5py
import pandas as pd
from network_properties import *
from plotting import *

def main(filename, output, indep_var, indep_var_name, dep_var_name):
    with h5py.File(filename, mode='r') as f:
        theta = f["theta"][:]
        phi = f["phi"][:]
        lon = (np.pi - theta)
        lat = (np.pi / 2 - phi)
        df = pd.DataFrame(columns=[indep_var_name, dep_var_name])
        for val in indep_var:
            print(f"{val}:")
            for k in set(f.keys()) - {"theta", "phi"}:
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
        #plot_line(output+"line.png", df)
        print("Creating histogram plot...")
        plot_hist(output+"hist.png", df)

main("../../data/euler/SWE_corr/CM_SWE_vorticity_PCC_s1_l24.h5", "../../data/euler/SWE_corr/",
     [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], "threshold", "distance (km)")
