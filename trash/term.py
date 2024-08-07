def mterm(lat_i, lat_j, lon_i, lon_j):
    """
        This function calculates the mixed term of the velocity induced by one point source at
        another point in a sphere.

        ==========================================================================================
        Parameters :
        ------------------------------------------------------------------------------------------
        Name    : Type [units]          Description
        ------------------------------------------------------------------------------------------
        lat_i : numpy.ndarray [deg]     Latitude point i.
        lat_j : numpy.ndarray [deg]     Latitude point j.
        lon_i : numpy.ndarray [deg]     Longitude point i.
        lon_j : numpy.ndarray [deg]     Longitude point j.
        ==========================================================================================
    """

    R = 1

    # Transform to polar and azimuthal angles in radians
    th_i, th_j = np.pi/2 - np.deg2rad(lat_i), np.pi/2 - np.deg2rad(lat_j)
    ph_i, ph_j = np.deg2rad(lon_i), np.deg2rad(lon_j)

    # Create 2D grids of latitudes and longitudes (i -> cst in rows, j -> cst in columns)
    gth_j, gth_i = np.meshgrid(th_j, th_i)
    gph_j, gph_i = np.meshgrid(ph_j, ph_i)

    # Calculate positions
    x_i = R*np.array([np.sin(gth_i)*np.cos(gph_i), np.sin(gth_i)*np.sin(gph_i), np.cos(gth_i)])
    x_j = R*np.array([np.sin(gth_j)*np.cos(gph_j), np.sin(gth_j)*np.sin(gph_j), np.cos(gth_j)])

    # Compute the cross product
    cross_product = np.empty_like(x_i)
    cross_product[0, :, :] = x_j[1, :, :] * x_i[2, :, :] - x_j[2, :, :] * x_i[1, :, :]
    cross_product[1, :, :] = x_j[2, :, :] * x_i[0, :, :] - x_j[0, :, :] * x_i[2, :, :]
    cross_product[2, :, :] = x_j[0, :, :] * x_i[1, :, :] - x_j[1, :, :] * x_i[0, :, :]

    # Compute the dot product
    dot_product = x_i[0, :, :] * x_j[0, :, :] + x_i[1, :, :] * x_j[1, :, :] + x_i[2, :, :] * x_j[2, :, :]

    # Calculate
    num = np.linalg.norm(cross_product, axis=0)
    denom = (R**2-dot_product)
    np.fill_diagonal(denom, np.inf)
    mterm = num/denom

    return mterm

def mixedtermA(lat, lon):
    R = 6.371e3

    # Transform to polar and azimuthal angles in radians
    th, ph = np.pi/2 - np.deg2rad(lat), np.deg2rad(lon)

    # Calculate positions
    x = R * np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])

    # Calculate inner product
    inner = R**2 - np.inner(x.T, x.T)
    np.fill_diagonal(inner, np.inf)

    # Calculate cross product
    cross = np.zeros((x.shape[1], x.shape[1], x.shape[0]))
    for i, a in enumerate(x.T):
        for j, b in enumerate(x.T):
            cross[i, j, :] = np.cross(a, b)
    cross = np.linalg.norm(cross, axis=2)

    # Calculate term
    return cross/inner