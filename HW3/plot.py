
np.save('Xtrain_xai', hi)
print(hi.shape)

print('1')
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))

contours = np.arange(268, 305, 4)

cb = plt.contourf(lons, lats, hi[100,:,:,0], #contours,
    transform=ccrs.PlateCarree(), cmap=get_cmap('inferno'))
print('2')

ax.coastlines()

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
    linewidth=1, color='dimgray', alpha=0.5, linestyle='--')

print('3')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabels_bottom = False
gl.ylabels_left = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([0, 90, 180, -90, 0])
gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
#gl.xlabel_style = {'size': 10, 'color': 'gray'}
#gl.ylabel_style = {'size': 10, 'color': 'gray'}

plt.subplots_adjust(bottom = 0.1,hspace=0.3,wspace=0.0)
print('4')

cbar = plt.colorbar(cb, orientation="horizontal")
cbar.set_label(r'Precip GF sensitivity',fontsize=12)

print('5')
#plt.show()
plt.savefig('one_year_test.png', dpi=300)
#plt.close()
