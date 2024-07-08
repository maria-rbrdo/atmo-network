import os
import numpy as np

def get_datazon(nlat, nt, job, fld, swend=False):
    """
    This function reads binary data from a specified file, processes it to
    account for the required shape, and optionally performs byte-swapping
    if the data is in a different endianness.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type              Description
    ------------------------------------------------------------------------------------------
    nlat    : int               Number of latitudes.
    nt      : int               Time-steps to consider.
    job     : str               Job name.
    fld     : str               Field name.
    swend   : bool, optional    Flag to indicate if byte-swapping is needed (default is False).
    ==========================================================================================

    Returns:
    -------
    numpy.ndarray
        2D array of the processed data, with shape (nlon/2, nlon).
    """

    # Define the directory
    datadir = os.path.expanduser("../../dataloc/")
    # datadir = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac")
    filepath = os.path.join(datadir, job, f"zavg.{fld}")

    # Open the file in binary read mode
    with open(filepath, 'rb') as f:
        # Read data from the file
        if swend:
            # If swend is True, read data and swap byte order
            data = np.fromfile(f, dtype='>f4', count=nlat * nt)  # big-endian float
        else:
            # Otherwise, read data directly
            data = np.fromfile(f, dtype='<f4', count=nlat * nt)  # little-endian float

    # Reshape the flat data array to a 2D array of shape (nlat,nt)
    data = data.reshape((nlat, nt))

    return data