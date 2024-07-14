import numpy as np
import os

def get_dataxy(nlon, tstr, job, fld, host='remotehost', swend=False):
    """
    This function reads binary data from a specified file, processes it to
    account for the required shape, and optionally performs byte-swapping
    if the data is in a different endianness.

    ==========================================================================================
    Parameters :
    ------------------------------------------------------------------------------------------
    Name    : Type              Description
    ------------------------------------------------------------------------------------------
    nlon    : int               Number of longitudes.
    tstr    : str               Time string.
    job     : str               Job name.
    fld     : str               Field name.
    host    : str               Hostname (localhost or remotehost)
    swend   : bool, optional    Flag to indicate if byte-swapping is needed (default is False).
    ==========================================================================================

    Returns:
    -------
    numpy.ndarray
        2D array of the processed data, with shape (nlon/2, nlon).
    """

    # Calculate the number of latitudes as half the number of longitudes
    nlat = nlon // 2

    # Define the directory
    if host == "localhost":
        datadir = os.path.expanduser("../../dataloc")
    elif host == "remotehost":
        datadir = os.path.abspath("/home/reboredoprad/bob/dataloc/bb/swvac")
    filepath = os.path.join(datadir, job, f"bobdata/{fld}.{tstr}")

    # Open the file in binary read mode
    with open(filepath, 'rb') as f:
        # Read data from the file
        if swend:
            # If swend is True, read data and swap byte order
            data = np.fromfile(f, dtype='>f4', count=-1)  # big-endian float
        else:
            # Otherwise, read data directly
            data = np.fromfile(f, dtype='<f4', count=-1)  # little-endian float

    # Reshape the flat data array to a 2D array of shape (nlat,nlon)
    data = data.reshape((nlat, nlon))

    # Print a confirmation message
    # print(f'got data: {fld}.{tstr}')

    # Return the processed data array
    return data
