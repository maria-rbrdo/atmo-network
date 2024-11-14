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

fields = ["pv"]
years = [2000, 2001, 2002, 2003, 2004, 2005]
DIN = "../../../data/ERA5/"
DOUT = "../../../output/ERA5/"

# Create folder to save files ......................................................................

if not os.path.exists(DOUT):
    os.makedirs(DOUT)

# Create files .....................................................................................

for yyyy in years:
    TT = 0

    # Create output file ...........................................................................
    FOUT = DOUT + f"Y{yyyy}-decfeb-daily-NH-850K"

    if os.path.exists(FOUT):
        print(f"File '{FOUT}' already exists, do you want to overwrite it? [y/n]")
        OVERWRITE = input()
        if OVERWRITE == "y":
            pass
        else:
            sys.exit()

    for i, mm in enumerate([12, 1, 2]):
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
            f.variables["valid_time"][:] / 24 - f.variables["valid_time"][0] / 24
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
                        store["time"][-time.shape[0],] = time + TT
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
