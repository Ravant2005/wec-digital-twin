import xarray as xr

FILE_PATH = "data/raw/era5_goa_jan2024.nc"


# Open the ERA5 NetCDF file
ds = xr.open_dataset(FILE_PATH)


print("\n========== DATASET ==========")
print(ds)


print("\n========== VARIABLES ==========")
for variable in ds.data_vars:
    print(variable)


print("\n========== COORDINATES ==========")
for coordinate in ds.coords:
    print(coordinate, ds[coordinate].values)


print("\n========== TIME ==========")
print("Start:", ds.time.values[0])
print("End:  ", ds.time.values[-1])
print("Number of time steps:", len(ds.time))


print("\n========== VARIABLE DETAILS ==========")

for variable in ds.data_vars:
    print(f"\n{variable}")
    print("  Long name:", ds[variable].attrs.get("long_name"))
    print("  Units:    ", ds[variable].attrs.get("units"))