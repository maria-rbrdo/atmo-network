import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scipy.signal import savgol_filter

tol = 1e-10
alpha = 0.3

deg = 1000  # no more than ten, weights problematic!
mu, w = roots_legendre(deg)
w = np.array(w)
mu = np.array(mu)

lat = np.rad2deg(np.arcsin(mu))
long = np.linspace(0, 360, 500)
Omega = 2 * np.pi / (3600 * 24)  # rad/s
H = 10*1e3  # m
R = 6.371*1e6  # m
g = 9.81  # m/s

Phi = np.zeros_like(lat)
zeta = np.zeros_like(lat)
u = np.zeros_like(lat)

qs = 0.2933 * 2 * Omega / H
qv = 2 * Omega / H + qs
qe = lambda phi: (qv * (50 < phi <= 90) + qs * (0 < phi <= 50)
                        + 2 * Omega * np.sin(np.deg2rad(phi)) / H * (-90 <= phi <= 0) + 0)
qe = np.vectorize(qe)
QE = qe(lat)
QE = np.vstack([np.atleast_2d(QE)]*500).T

# Drag
r0 = 1/20
r = lambda phi: r0 * (1 - np.tanh((np.deg2rad(phi)-np.pi/18)/(np.pi/18)))
r = np.vectorize(r)
R = r(lat)
R = np.vstack([np.atleast_2d(R)]*500).T

# Topography
phi = np.linspace(-np.pi / 2, np.pi / 2, 250)  # rad
lamb = np.linspace(0, 2*np.pi, 500)  # rad
Phi, Lamb = np.meshgrid(phi, lamb)

H = 10000  # m
A0 = 0.15*H
tau = 100  # days1
Omega = np.deg2rad(360)  # rad/day
w0 = -0.000
hb = lambda phi, lamb: A0*np.sin(2*phi)**2*np.cos(2*lamb) * (phi >= 0)

HB = hb(Phi, Lamb)

# Plot
fig = plt.figure(figsize=(5, 4))
plt.rcParams.update({'font.size': 18})

ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 40))
ax.set_global()
ax.gridlines(linestyle=":", color="white")

x = np.rad2deg(lamb)
y = np.rad2deg(phi)
filled_c = ax.contourf(x, y, HB[:, :].T, transform=ccrs.PlateCarree(), cmap=sns.color_palette("Spectral_r",
                                                                                                        as_cmap=True))
ax.contour(x, y, HB[:, :].T, colors='black', transform=ccrs.PlateCarree(), linewidths=1.5)

#filled_c = ax.contourf(long, lat, QE[:, :]*H/Omega, transform=ccrs.PlateCarree(), cmap=sns.color_palette(
# "Spectral_r", as_cmap=True))
#ax.contour(long, lat, QE[:, :]*H/Omega, colors='black', transform=ccrs.PlateCarree(), linewidths=1.5)

#filled_c = ax.contourf(long, lat, R[:, :], transform=ccrs.PlateCarree(), cmap=sns.color_palette("Spectral_r",
# as_cmap=True))
#ax.contour(long, lat, R[:, :], colors='black', transform=ccrs.PlateCarree(), linewidths=1.5)

cbar = fig.colorbar(filled_c, orientation='vertical', extend="both", pad=0.1, aspect=10)
cbar.set_label(r"$h_b$ (m)")
#cbar.set_label(r"$q_e H / \Omega$ (unitless)")
#cbar.set_label(r"$\beta$ (d$^{-1}$)")
plt.subplots_adjust(left=0.025, right=0.8, top=0.9, bottom=0.1)
fig.savefig("settings.png", dpi=300)