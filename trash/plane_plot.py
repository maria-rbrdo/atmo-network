# sns.heatmap(np.flip(matrix.T, 0), ax=ax, cmap=my_cmap, center=my_center,
#            cbar_kws=dict(use_gridspec=False, location="top", aspect=60, extend='both',
#                          label=f"{measure}", pad=0.01), transform=ccrs.PlateCarree())
# x_ticks = 9
# y_ticks = 5
# ax.set_xticks(np.linspace(0, len(np.unique(lon)), x_ticks))
# ax.set_xticklabels(np.linspace(-180, 180, x_ticks, dtype=int))
# ax.set_yticks(np.linspace(0, len(np.unique(lat)), y_ticks))
# ax.set_yticklabels(np.linspace(90, -90, y_ticks, dtype=int))

# ax.set_xlabel('longitude (deg)')
# ax.set_ylabel('latitude (deg)')