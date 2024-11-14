import warnings
from calendar import monthrange

import cdsapi

warnings.filterwarnings("ignore")

# Functions ......................................................................................

def build_request(yyyy, mm):
    """
    This function builds a request for ECMWF's Metereological Archival and Retrieval System.
    It is set for daily measures of potential vorticity, velocity, and relative vorticity
    at 00:00:00 throughout a whole month in the NH.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type              Description
    ------------------------------------------------------------------------------------------
    yyyy    : int               Year to retreive.
    mm      : int               Month to retreive.
    ==========================================================================================

    Returns:
    -------
    numpy.ndarray
        Adjacency matrix, with shape (nlon*nlat, nlon*nlat).
    """
    dd = monthrange(yyyy, mm)[-1]
    request = {
        "date": f"{yyyy}-{mm:02d}-01/to/{yyyy}-{mm:02d}-{dd:02d}",
        "expver": "1",
        "levelist": "850",
        "levtype": "pt",
        "param": "60.128/131/132/138.128",  # Info at https://apps.ecmwf.int/codes/grib/param-db/
        "stream": "oper",  # Denotes ERA5. Ensemble members are selected by 'enda'
        "time": "00:00:00",
        "type": "an",
        "area": "90/-180/0/180",  # North, West, South, East. Default: global
        "grid": "1.0/1.0",  # Latitude/longitude
        "format": "netcdf",  # Output needs to be regular lat-lon, so only works with 'grid'!
    }
    return request


# Parameters .......................................................................................

yarr = [2005, 2006, 2007, 2008, 2009, 2010]

# Execution ........................................................................................

DATASET = "reanalysis-era5-complete"
c = cdsapi.Client()
for i, year in enumerate(yarr):
    if i == 0:
        marr = [12]
    elif i == len(yarr) - 1:
        marr = [1, 2]
    else:
        marr = [1, 2, 12]

    for month in marr:
        print(
            f""" 
                ***Loading Data*** 
                {year}-{month:02d}-01/to/{year}-{month:02d}-{monthrange(year, month)[-1]} 
                """
        )
        r = build_request(year, month)
        c.retrieve(DATASET, r, f"Y{year}M{month:02d}-daily-NH-850K.nc")
