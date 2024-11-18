"""
====================================================================================================
NETCDF Data Handling Script
====================================================================================================
This script reads a netcdf file and stores it in an h5 file.
----------------------------------------------------------------------------------------------------
"""

import os
import sys

import h5py
import netCDF4

# Parameters .......................................................................................

fields = ["latitude", "longitude", "time", "pv", "u", "v", "vo"]
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
DIN = "../../../data/ERA5/"
DOUT = "../../../output/ERA5/"

# Create files .....................................................................................

for yyyy in years:
    TT = 0

    # Create output folder and file ................................................................
    DOUTT = DOUT + f"Y{yyyy}-DJFM-daily-NH-850K/data/"
    FOUT = DOUTT + f"Y{yyyy}-DJFM-daily-NH-850K.h5"

    if not os.path.exists(DOUTT):
        os.makedirs(DOUTT)

    if os.path.exists(FOUT):
        print(f"File '{FOUT}' already exists, do you want to overwrite it? [y/n]")
        OVERWRITE = input()
        if OVERWRITE == "y":
            pass
        else:
            continue

    for i, mm in enumerate([12, 1, 2, 3]):
        # Get input file ........................................................................
        if mm == 12:
            f = netCDF4.Dataset(
                DIN + f"Y{yyyy-1}M{mm:02d}-daily-NH-850K.nc", "r", format="NETCDF4"
            )
        else:
            f = netCDF4.Dataset(
                DIN + f"Y{yyyy}M{mm:02d}-daily-NH-850K.nc", "r", format="NETCDF4"
            )

        # Store data .............................................................................
        time = (
            f.variables["valid_time"][:] / 24 / 3600
            - f.variables["valid_time"][0] / 24 / 3600
        )  # days since 1900-01-01
        with h5py.File(FOUT, mode="a") as store:
            for field in fields:
                if i == 0 and field in store.keys():
                    del store[field]

                if field in ["latitude", "longitude"]:
                    if i == 0:
                        store.create_dataset(field, data=f.variables[field][:])
                elif field == "time":
                    if i == 0:
                        store.create_dataset(
                            "time", chunks=True, data=time, maxshape=(None,)
                        )
                    else:
                        store["time"].resize(
                            (store["time"].shape[0] + time.shape[0]), axis=0
                        )
                        store["time"][-time.shape[0] :,] = time + TT
                else:
                    data = f.variables[field][:]
                    if i == 0:
                        store.create_dataset(
                            field,
                            chunks=True,
                            data=data.transpose(1, 2, 0),
                            maxshape=(data.shape[1], data.shape[2], None),
                        )
                    else:
                        store[field].resize(
                            (store[field].shape[2] + data.shape[0]), axis=2
                        )
                        store[field][:, :, -data.shape[0] :] = data.transpose(1, 2, 0)
            f.close()
        # Increase total time considered ........................................................
        TT += time.shape[0]
