import cartopy.crs as ccrs
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(-np.pi / 2, np.pi / 2, 250)  # rad
lamb = np.linspace(0, 2*np.pi, 500)  # rad
t = np.arange(25, 200, 25)  # days
Phi, Lamb, T = np.meshgrid(phi, lamb, t, indexing='ij')

H = 10000  # m
A0 = 0.15*H
tau = 100  # days1
Omega = np.deg2rad(360)  # rad/day
w0 = -0.000
A = lambda t: A0 * np.cos(np.pi/2 * (1-t/tau))**2 * (t < tau) + A0 * (t >= tau)
hb = lambda phi, lamb, t: A(t)*np.sin(2*phi)**2*np.cos(2*lamb + w0*t) * (phi >= 0)

HB = hb(Phi, Lamb, T)

# Plotting
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(15, 4))

x = np.rad2deg(lamb)
y = np.rad2deg(phi)

for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, projection=ccrs.Orthographic(0, 40))
    ax.set_global()
    filled_c = ax.contourf(x, y, HB[:, :, i], transform=ccrs.PlateCarree(), cmap = sns.color_palette("Spectral_r", as_cmap=True), vmin = -A0, vmax= A0)
    ax.gridlines(linestyle=":", color="white")
    ax.contour(x, y, HB[:, :, i], colors='black', transform=ccrs.PlateCarree(), linewidths=1.5)
    ax.set_title(f'{t[i]:.1f} d')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(filled_c, cax=cbar_ax, orientation='vertical', extend="both")
cbar.set_label(r"$h_b$ (m)")
fig.savefig("forcing", dpi=300, bbox_inches='tight')
plt.close()