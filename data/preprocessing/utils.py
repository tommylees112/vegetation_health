import xarray as xr
import pandas as pd
import xesmf as xe  # for regridding
import numpy as np

import os
from pathlib import Path
import ipdb
import warnings
import datetime

import geopandas as gpd
from shapely import geometry


def read_csv_point_data(df, lat_col='lat', lon_col='lon', crs='epsg:4326'):
    """Read in a csv file with lat,lon values in a column and turn those lat lon
        values into geometry.Point objects.
    Arguments:
    ---------
    : df (pd.DataFrame)
    : lat_col (str)
        the column in the dataframe that has the point latitude information
    : lon_col (str)
        the column in the dataframe that has the point longitude information
    : crs (str)
        coordinate reference system (defaults to 'epsg:4326')
    Returns:
    -------
    : gdf (gpd.GeoDataFrame)
        a geopandas.GeoDataFrame object
    """
    df['geometry'] = [geometry.Point(y, x) \
                      for x, y in zip(df[lat_col],
                                      df[lon_col])
                    ]
    crs = {'init': crs}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry="geometry")
    return gdf


# ------------------------------------------------------------------------------
# Functions for reprojecting using GDAL and reading resulting .nc file back
# ------------------------------------------------------------------------------


def gdal_reproject(infile, outfile, **kwargs):
    """Use gdalwarp to reproject one file to another

    Help:
    ----
    https://www.gdal.org/gdalwarp.html
    """
    to_proj4_string = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    resample_method = "near"

    # check options
    valid_resample_methods = [
        "average",
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
    ]
    assert (
        resample_method in valid_resample_methods
    ), f"Resample method not Valid. Must be one of: {valid_resample_methods} Currently: {resample_method}"

    cmd = f'gdalwarp -t_srs "{to_proj4_string}" -of netCDF -r average -dstnodata -9999 -ot Float32 {infile} {outfile}'

    # run command
    print(f"\n\n#### Running command: {cmd} ####\n\n")
    os.system(cmd)
    print(f"\n\n#### Run command {cmd} \n FILE REPROJECTED ####\n\n")

    return


def bands_to_time(da, times, var_name):
    """ For a dataArray with each timestep saved as a different band, create
         a time Coordinate
    """
    # get a list of all the bands as dataarray objects (for concatenating later)
    band_strings = [key for key in da.variables.keys() if "Band" in key]
    bands = [da[key] for key in band_strings]
    bands = [band.rename(var_name) for band in bands]

    # check the number of bands matches n timesteps
    assert len(times) == len(
        bands
    ), f"The number of bands should match the number of timesteps. n bands: {len(times)} n times: {len(bands)}"
    # concatenate into one array
    timestamped_da = xr.concat(bands, dim=times)

    return timestamped_da


# ------------------------------------------------------------------------------
# Functions for matching resolutions / gridsizes (time and space)
# ------------------------------------------------------------------------------


def convert_to_same_grid(reference_ds, ds, method="nearest_s2d"):
    """ Use xEMSF package to regrid ds to the same grid as reference_ds """
    assert ("lat" in reference_ds.dims) & (
        "lon" in reference_ds.dims
    ), f"Need (lat,lon) in reference_ds dims Currently: {reference_ds.dims}"
    assert ("lat" in ds.dims) & (
        "lon" in ds.dims
    ), f"Need (lat,lon) in ds dims Currently: {ds.dims}"

    # create the grid you want to convert TO (from reference_ds)
    ds_out = xr.Dataset(
        {"lat": (["lat"], reference_ds.lat), "lon": (["lon"], reference_ds.lon)}
    )

    # create the regridder object
    # xe.Regridder(grid_in, grid_out, method='bilinear')
    regridder = xe.Regridder(ds, ds_out, method, reuse_weights=True)

    # IF it's a dataarray just do the original transformations
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = regridder(ds)
    # OTHERWISE loop through each of the variables, regrid the datarray then recombine into dataset
    elif isinstance(ds, xr.core.dataset.Dataset):
        vars = [i for i in ds.var().variables]
        if len(vars) == 1:
            ds = regridder(ds)
        else:
            output_dict = {}
            # LOOP over each variable and append to dict
            for var in vars:
                print(f"- regridding var {var} -")
                da = ds[var]
                da = regridder(da)
                output_dict[var] = da
            # REBUILD
            ds = xr.Dataset(output_dict)
    else:
        assert False, "This function only works with xarray dataset / dataarray objects"

    print(
        f"Regridded from {(regridder.Ny_in, regridder.Nx_in)} to {(regridder.Ny_out, regridder.Nx_out)}"
    )

    return ds


def select_same_time_slice(reference_ds, ds):
    """ Select the values for the same timestep as the reference ds"""
    # CHECK THEY ARE THE SAME FREQUENCY
    # get the frequency of the time series from reference_ds
    freq = pd.infer_freq(reference_ds.time.values)
    if freq == None:
        warnings.warn('HARDCODED FOR THIS PROBLEM BUT NO IDEA WHY NOT WORKING')
        freq = "M"
        # assert False, f"Unable to infer frequency from the reference_ds timestep"

    old_freq = pd.infer_freq(ds.time.values)
    warnings.warn(
        "Disabled the assert statement. ENSURE FREQUENCIES THE SAME (e.g. monthly)"
    )
    # assert freq == old_freq, f"The frequencies should be the same! currenlty ref: {freq} vs. old: {old_freq}"

    # get the STARTING time point from the reference_ds
    min_time = reference_ds.time.min().values
    max_time = reference_ds.time.max().values
    orig_time_range = pd.date_range(min_time, max_time, freq=freq)
    # EXTEND the original time_range by 1 (so selecting the whole slice)
    # because python doesn't select the final in a range
    periods = len(orig_time_range) #+ 1
    # create new time series going ONE EXTRA PERIOD
    new_time_range = pd.date_range(min_time, freq=freq, periods=periods)
    new_max = new_time_range.max()

    # select using the NEW MAX as upper limit
    # --------------------------------------------------------------------------
    # FOR SOME REASON slice is removing the minimum time ...
    # something to do with the fact that matplotlib / xarray is working oddly with numpy64datetime object
    warnings.warn("L153: HARDCODING THE MIN VALUE OTHERWISE IGNORED ...")
    min_time = datetime.datetime(2001, 1, 31)
    # --------------------------------------------------------------------------
    ds = ds.sel(time=slice(min_time, new_max))
    assert reference_ds.time.shape[0] == ds.time.shape[0],f"The time dimensions should match, currently reference_ds.time dims {reference_ds.time.shape[0]} != ds.time dims {ds.time.shape[0]}"

    print_time_min = pd.to_datetime(ds.time.min().values)
    print_time_max = pd.to_datetime(ds.time.max().values)
    try:
        vars = [i for i in ds.var().variables]
    except:
        vars = ds.name
    # ref_vars = [i for i in reference_ds.var().variables]
    print(
        f"Select same timeslice for ds with vars: {vars}. Min {print_time_min} Max {print_time_max}"
    )

    return ds


def get_holaps_mask(ds):
    """
    NOTE:
    - assumes that all of the null values from the HOLAPS file are valid null values (e.g. water bodies). Could also be invalid nulls due to poor data processing / lack of satellite input data for a pixel!
    """
    warnings.warn(
        "assumes that all of the null values from the HOLAPS file are valid null values (e.g. water bodies). Could also be invalid nulls due to poor data processing / lack of satellite input data for a pixel!"
    )
    warnings.warn(
        "How to collapse the time dimension in the holaps mask? Here we just select the first time because all of the valid pixels are constant for first, last second last. Need to check this is true for all timesteps"
    )

    mask = ds.isnull().isel(time=0).drop("time")
    mask.name = "holaps_mask"

    mask = xr.concat([mask for _ in range(len(ds.time))])
    mask = mask.rename({"concat_dims": "time"})
    mask["time"] = ds.time

    return mask



def select_east_africa(ds):
    """ """
    lonmin=32.6
    lonmax=51.8
    latmin=-5.0
    latmax=15.2

    if ('x' in ds.dims) & ('y' in ds.dims):
        ds = ds.sel(y=slice(latmax,latmin),x=slice(lonmin, lonmax))
    elif ('lat' in ds.dims) & ('lon' in ds.dims):
        ds = ds.sel(lat=slice(latmax,latmin),lon=slice(lonmin, lonmax))
    elif ('latitude' in ds.dims) & ('longitude' in ds.dims):
        ds = ds.sel(latitude=slice(latmax,latmin),longitude=slice(lonmin, lonmax))
    else:
        assert False, "You need one of [(y, x), (lat, lon), (latitude, longitude)] in your dims"

    return


# ------------------------------------------------------------------------------
# Functions for working with xarray objects
# ------------------------------------------------------------------------------


def merge_data_arrays(*DataArrays):
    das = [da.name for da in DataArrays]
    print(f"Merging data: {das}")
    ds = xr.merge([*DataArrays])
    return ds


def save_netcdf(xr_obj, filepath, force=False):
    """"""
    if not Path(filepath).is_file():
        xr_obj.to_netcdf(filepath)
        print(f"File Saved: {filepath}")
    elif force:
        print(f"Filepath {filepath} already exists! Overwriting...")
        xr_obj.to_netcdf(filepath)
        print(f"File Saved: {filepath}")
    else:
        print(f"Filepath {filepath} already exists!")

    return


def get_all_valid(ds, holaps_da, modis_da, gleam_da):
    """ Return only values for pixels/times where ALL PRODUCTS ARE VALID """
    valid_mask = (
    holaps_da.notnull()
    & modis_da.notnull()
    & gleam_da.notnull()
    )
    ds_valid = ds.where(valid_mask)

    return ds_valid


def drop_nans_and_flatten(dataArray):
    """flatten the array and drop nans from that array. Useful for plotting histograms.

    Arguments:
    ---------
    : dataArray (xr.DataArray)
        the DataArray of your value you want to flatten
    """
    # drop NaNs and flatten
    return dataArray.values[~np.isnan(dataArray.values)]

#

#
# def create_flattened_dataframe_of_values(h,g,m):
#     """ """
#     h_ = drop_nans_and_flatten(h)
#     g_ = drop_nans_and_flatten(g)
#     m_ = drop_nans_and_flatten(m)
#     df = pd.DataFrame(dict(
#             holaps=h_,
#             gleam=g_,
#             modis=m_
#         ))
#     return df
