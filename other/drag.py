import cartopy.crs as ccrs
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(-np.pi / 2, np.pi / 2, 250)  # rad
lamb = np.linspace(0, 2*np.pi, 500)  # rad
t = np.arange(0, 200, 25)  # days
Phi, Lamb = np.meshgrid(phi, lamb, indexing='ij')

r0 = 1/20 # days
r = lambda phi, lamb: r0 * (1 - np.tanh((phi-np.pi/18)/(np.pi/18)))

# Plotting
plt.rcParams.update({'font.size': 30})
fig = plt.figure(figsize=(10, 10))

plt.plot(r(phi, lamb), np.rad2deg(phi), "k", linewidth=5)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.xlabel(r"Damping rate (days$^{−1}$)")
plt.ylabel("Latitude (deg)")
plt.ylim([-90, 90])
# R = r(Phi, Lamb)
# x = np.rad2deg(lamb)
# y = np.rad2deg(phi)

# ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIV())
# ax.set_global()
# filled_c = ax.contourf(x, y, R, 50, transform=ccrs.PlateCarree(), cmap="rocket")
# ax.contour(x, y, R, levels=filled_c.levels, colors='black', transform=ccrs.PlateCarree())

# fig.colorbar(filled_c, orientation='vertical')

fig.savefig("drag", dpi=200, bbox_inches='tight')
plt.close()