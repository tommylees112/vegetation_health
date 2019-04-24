"""
@tommylees112

An awful script (sorry Gabi) the data is read in separately for each product.
They are then preprocessed:
    resample_time
    select_time_slice
    regrid_to_reference

using the precipitation data as reference.
"""
# test.py
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

from preprocessing.utils import (
    gdal_reproject,
    bands_to_time,
    convert_to_same_grid,
    select_same_time_slice,
    save_netcdf,
    get_holaps_mask,
    merge_data_arrays,
)


def correct_time_slice(ds, reference_ds):
    """select the same time slice as the reference data"""
    correct_time_slice = select_same_time_slice(reference_ds, ds)

    return correct_time_slice


def resample_time(ds, resample_str="M"):
    """ should resample to the given timestep """
    resampled_time_data = ds.resample(time=resample_str).first()
    return resampled_time_data


def regrid_to_reference(ds, reference_ds, method="nearest_s2d"):
    """ regrid data (spatially) onto the same grid as reference data """

    regrid_data = convert_to_same_grid(
        reference_ds, ds, method=method
    )

    return regrid_data


def use_reference_mask(ds, mask, one_time=False):
    # if only one timestep (e.g. landcover) then convert to one time
    if one_time:
        self.mask = self.mask.isel(time=0)

    masked_d = ds.where(~mask.values)
    return masked_d


def rename_lat_lon(ds):
    rename_latlon = ds.rename({"longitude": "lon", "latitude": "lat"})
    return rename_latlon


def select_time_slice(ds, timemin, timemax):
    return ds.sel(time=slice(timemin, timemax))


# ------------------------------------------------------------------------------
# Read the data
# ------------------------------------------------------------------------------
DATA_DIR1 = Path("/soge-home/users/chri4118/EA_data")
DATA_DIR2 = Path("/soge-home/projects/crop_yield/EGU_compare")

et = xr.open_dataset(DATA_DIR1 / "ET_EastAfrica.nc")
lst = xr.open_dataset(DATA_DIR1 / "LST_EastAfrica.nc")[["lst_day", "lst_night"]]
sm = xr.open_dataset(DATA_DIR1 / "SM_EastAfrica.nc")[['sm','sm_uncertainty']]
ndvi = xr.open_dataset(DATA_DIR1 / "NDVI_EastAfrica.nc")[["ndvi", "evi"]]
precip = xr.open_dataset(DATA_DIR2 / "EA_chirps_monthly.nc")

# ------------------------------------------------------------------------------
# Clean the data (same timesteps and same gridsizes)
# ------------------------------------------------------------------------------
# RESAMPLE THE REFERENCE DATA
precip = select_time_slice(precip, '2000-02-14','2016-12-01')
precip = resample_time(precip)
precip = rename_lat_lon(precip)
reference_ds = precip

all_vars = [et,lst,sm,ndvi,precip]
names = ["et","lst","sm","ndvi","precip"]

# RESAMPLE data (except 'precip')
out = []
for ix, ds in enumerate(all_vars[:-1]):
    name = names[ix]
    print(f"\n*** working on ds: {name} ***")
    # select same time slice
    ds = resample_time(ds)
    ds = select_time_slice(ds, '2000-02-14','2016-12-01')
    # ds = correct_time_slice(ds, reference_ds)
    print("selected same time slice")
    # convert to same grid
    ds = regrid_to_reference(ds, reference_ds)
    print("converted to same grid")
    out.append(ds)

# ------------------------------------------------------------------------------
# Merge all of the datasets
# ------------------------------------------------------------------------------
alldata = out + [precip]
OUT = xr.merge(alldata)

# ------------------------------------------------------------------------------
# Save the data to netcdf format
# ------------------------------------------------------------------------------
OUT.to_netcdf(DATA_DIR1 / "OUT2.nc")
OUT.to_netcdf(DATA_DIR2 / "predict_vegetation_health.nc")
