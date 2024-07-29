"""
========================================================================================================================
NETCDF Data Handling Script
========================================================================================================================
This script reads a netcdf file and stores it in an h5 file.
------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
from scipy.io import netcdf_file

# Parameters ...........................................................................................................
fpath = "../../dataloc/netcdf/netcdf_data.nc"
host = "localhost"

# Host .................................................................................................................
if host == "localhost":
    folder = os.path.expanduser("../../dataloc/netcdf/netdata/")
    file = os.path.expanduser(f"../../dataloc/netcdf/netdata/data")
elif host == "remotehost":
    folder = os.path.abspath("/home/reboredoprad/bob/dataloc/netcdf/netdata/")
    file = os.path.abspath("/home/reboredoprad/bob/dataloc/netcdf/netdata/data")

# Create folder to save files ..........................................................................................
if not os.path.exists(folder):
    os.mkdir(folder)
try:
    os.remove(file)
    print(f"Previous file '{file}' deleted successfully.")
except:
    pass

# Get data .............................................................................................................
f = netcdf_file(fpath, 'r', mmap=False)
lat = f.variables['latitude'][:]
lon = f.variables['longitude'][:]
time = f.variables['time'][:]
pv = f.variables['pv'][:]

# Store data ...........................................................................................................

with h5py.File(file, mode='a') as store:
    store.create_dataset("data", data=pv)
    store.create_dataset("longitude", data=lon)
    store.create_dataset("latitude", data=lat)
    store.create_dataset("time", data=time)

f.close()
