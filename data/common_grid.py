"""
File for putting netcdf (.nc) files onto a common grid. This means that the
 location, timesteps and resolution of the data is now the same. This makes
 working with the data a lot simpler!

NOTES:
- vars_list : all of the variables that we are regridding onto a common grid.
               They are all from different sources
    [lst_day, lst_night, lst_mean, lst_mean, evap, baresoil_evap, pet, transp,
    surface_sm, rootzone_sm, sm, precip, ndvi, evi]

- East Africa is defined here as the area of the original .nc file (spi_spei.nc)
    lat min - lat max : -4.9750023 15.174995
    lon min - lon max : 32.524994 48.274994
    BoundingBox(left, bottom, right, top)
        (32.524994, -4.9750023, 15.174995, 48.274994)

- Time Range
    2010-01-01 : 2017-01-01
"""

import xarray as xr
import xesmf as xe # for regridding
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tqdm # for progress-bars
import click
import warnings

from drought_masking import create_drought_mask
from utils import save_netcdf


def pickle_obj(obj, filename):
    """ write to pickle
    """
    assert (filename.split('.')[-1] == "pickle") or (filename.split('.')[-1] == "pkl"), f"filename should end with ('pickle','pkl') currently: {filename}"
    output = open(filename, 'wb')
    pickle.dump(obj, output)

    return


def read_pickle_df(filepath):
  """ read pickled object
  """
  obj = pd.read_pickle(filepath)
  return obj


def normalise_coordinate_names(ds):
    """ rename latitude/longitude to lat/lon """
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude":"lat"})
        print("Renamed `latitude` to `lat`")
    if "longitude" in ds.dims:
        ds = ds.rename({"longitude":"lon"})
        print("Renamed `longitude` to `lon`")
    assert "time" in ds.dims, f"Must have `time` as dimension in object. Currently: {ds.dims}"

    return ds


def select_same_time_slice(reference_ds, ds):
    """ Select the values for the same timestep as the """
    # CHECK THEY ARE THE SAME FREQUENCY
    # get the frequency of the time series from reference_ds
    freq = pd.infer_freq(reference_ds.time.values)
    old_freq = pd.infer_freq(ds.time.values)
    assert freq == old_freq, f"The frequencies should be the same! currenlty ref: {freq} vs. old: {old_freq}"

    # get the STARTING time point from the reference_ds
    min_time = reference_ds.time.min().values
    max_time = reference_ds.time.max().values
    orig_time_range = pd.date_range(min_time, max_time, freq=freq)
    # EXTEND the original time_range by 1 (so selecting the whole slice)
    # because python doesn't select the final in a range
    periods = len(orig_time_range) + 1
    # create new time series going ONE EXTRA PERIOD
    new_time_range = pd.date_range(min_time, freq=freq, periods=periods)
    new_max = new_time_range.max()

    # select using the NEW MAX as upper limit
    ds = ds.sel(time=slice(min_time, new_max))
    # assert reference_ds.time.shape[0] == ds.time.shape[0],"The time dimensions should match, currently reference_ds.time dims {reference_ds.time.shape[0]} != ds.time dims {ds.time.shape[0]}"

    print_time_min = pd.to_datetime(ds.time.min().values)
    print_time_max = pd.to_datetime(ds.time.max().values)
    try:
        vars = [i for i in ds.var().variables]
    except:
        vars = ds.name
    ref_vars = [i for i in reference_ds.var().variables]
    print(f"Select same timeslice for ds with vars: {vars}. Min {print_time_min} Max {print_time_max}")

    return ds


def select_same_lat_lon_slice(reference_ds, ds):
    """
    Take a slice of data from the `ds` according to the bounding box from
     the reference_ds.
    NOTE: - latitude has to be from max() to min() for some reason?
          - becuase it's crossing the equator? e.g. -14.:8.
    Therefore, have to run an if statement to decide which way round to put the data
    """
    # lat_bounds = [reference_ds.lat.min(),reference_ds.lat.max()]
    # lon_bounds = [reference_ds.lon.min(),reference_ds.lon.max()]
    if len(ds.sel(lat=slice(reference_ds.lat.min(), reference_ds.lat.max())).lat) == 0:
        ds = ds.sel(lat=slice(reference_ds.lat.max(), reference_ds.lat.min()))
    else:
        ds = ds.sel(lat=slice(reference_ds.lat.min(), reference_ds.lat.max()))
    ds = ds.sel(lon=slice(reference_ds.lon.min(), reference_ds.lon.max()))

    try:
        vars = [i for i in ds.var().variables]
    except:
        vars = ds.name
    ref_vars = [i for i in reference_ds.var().variables]
    print(f"Select the same bounding box for ds {vars} from reference_ds {ref_vars}")
    return ds


def open_drought_ds(data_path = 'spei_spi.nc'):
    """ returns the raw dataset & 2x boolean masks (SPI/SPEI)"""
    # Data path on MONTHLY GRID

    # open the data ()
    ds = xr.open_dataset(data_path)
    ds = ds.rename({"value":"spi"})
    ds = normalise_coordinate_names(ds)

    # turn the values less than -1 into mask
    spei_drought = ds.spei.where(ds.spei < -1)
    spi_drought = ds.spi.where(ds.spi < -1)

    # turn into a boolean mask
    drought_spei = spei_drought.notnull()
    drought_spei = drought_spei.rename('drought_spei')
    drought_spi = spi_drought.notnull()
    drought_spi = drought_spi.rename('drought_spi')

    return ds, drought_spei, drought_spi


def ensure_same_time(reference_ds, ds):
    """ convert the TIMESTEPS to the same values
        e.g. • if Monthly data sometimes its in the middle - 01-16-98
             • other times its the start 01-01-98, othertimes 01-31-98
        Set them all to the same frequency using the freq from reference_ds
     """
    freq = pd.infer_freq(reference_ds.time.values)
    dr = pd.date_range(reference_ds.time.min().values , periods=len(ds.time.values), freq=freq)
    ds['time'] = dr

    return ds


def convert_to_same_time_freq(reference_ds,ds):
    """ Upscale or downscale data so on the same time frequencies
    e.g. convert daily data to monthly ('MS' = month start)
    """
    freq = pd.infer_freq(reference_ds.time.values)
    ds = ds.resample(time='MS').median(dim='time')

    try:
        vars = [i for i in ds.var().variables]
    except:
        vars = ds.name
    print(f"Resampled ds ({vars}) to {freq}")
    return ds


def convert_to_same_grid(reference_ds, ds):
    """ Use xEMSF package to regrid ds to the same grid as reference_ds """
    assert ("lat" in reference_ds.dims)&("lon" in reference_ds.dims), f"Need (lat,lon) in reference_ds dims Currently: {reference_ds.dims}"
    assert ("lat" in ds.dims)&("lon" in ds.dims), f"Need (lat,lon) in ds dims Currently: {ds.dims}"

    # create the grid you want to convert TO (from reference_ds)
    ds_out = xr.Dataset({
        'lat': (['lat'], reference_ds.lat),
        'lon': (['lon'], reference_ds.lon),
    })

    # create the regridder object
    # xe.Regridder(grid_in, grid_out, method='bilinear')
    regridder = xe.Regridder(ds, ds_out, 'bilinear', reuse_weights=True)

    # IF it's a dataarray just do the original transformations
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = regridder(ds)
    # OTHERWISE loop through each of the variables, regrid the datarray then recombine into dataset
    elif isinstance(ds, xr.core.dataset.Dataset):
        vars = [i for i in ds.var().variables]
        if len(vars) ==1 :
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

    print(f"Regridded from {(regridder.Ny_in, regridder.Nx_in)} to {(regridder.Ny_out, regridder.Nx_out)}")

    return ds


def netcdf_to_same_dim_shapes(reference_ds, *other_netcdfs):
    """ loop over each of the other_netcdfs and reshape them in TIME, LAT/LON
    to be the same as the reference_ds.
    This allows them to be stored as a single netcdf file
    """
    ds_to_merge = []
    for ds_ in tqdm.tqdm([*other_netcdfs]):
        ds_ = normalise_coordinate_names(ds_)
        assert ("lat" in ds_.dims)&("lon" in ds_.dims),f"lat and lon should be in ds.dims. Currently: {ds_.dims}"

        # select the same SLICE of time (reference_ds.min() - reference_ds.min())
        ds_ = convert_to_same_time_freq(reference_ds, ds_)
        ds_ = select_same_time_slice(reference_ds, ds_)

        # REGRID to same dimensions
        ds_ = convert_to_same_grid(reference_ds, ds_)
        # select the same lat lon slice
        ds_ = select_same_lat_lon_slice(reference_ds, ds_)

        # ENSURE DIMS MATCH
        first_var = [i for i in reference_ds.var().variables][0]
        np.testing.assert_allclose(reference_ds[first_var].shape[0], ds_.shape[0], atol=1), f"The TIME DIMENSION should be equal. Currently (time, lat, lon) {ds_.shape} should be {reference_ds[first_var].shape}"
        assert reference_ds[first_var].shape[1] == ds_.shape[1], f"The LAT DIMENSION should be equal. Currently (time, lat, lon) {ds_.shape} should be {reference_ds[first_var].shape}"
        assert reference_ds[first_var].shape[2] == ds_.shape[2], f"The LON DIMENSION should be equal. Currently (time, lat, lon) {ds_.shape} should be {reference_ds[first_var].shape}"

        ds_to_merge.append(ds_)

    return ds_to_merge


def get_all_vars_from_ds(ds):
    """ return a list of strings for all variables in dataset """
    assert isinstance(ds, xr.core.dataset.Dataset), f"Currently only works with xr.Dataset objects. ds = {type(ds)}"
    vars = [i for i in ds.var().variables]
    return vars


def append_reference_ds_vars(reference_ds, ds_to_merge):
    """merge in the reference_ds variables to the ds_to_merge"""
    for var in reference_ds.var().variables:
        ds_to_merge.append(reference_ds[var])

    return ds_to_merge


def merge_netcdfs_to_one_file(reference_ds, *other_netcdfs, drought=False):
    """ merge the netcdf files with multiple variables into ONE netcdf file.
    Use the structure from the reference_ds to get the same TIME & LAT/LON GRID.
    """
    ds_to_merge = netcdf_to_same_dim_shapes(reference_ds, *other_netcdfs)
    ds_to_merge = append_reference_ds_vars(reference_ds, ds_to_merge)

    # join in the drought mask
    # --------------------------------------------------------------------------
    warnings.warn('This is done once here to put the ndvi mask into the nc file')
    output_ds = xr.merge(ds_to_merge)
    # --------------------------------------------------------------------------
    if drought:
        warnings.warn('Should be less focused on hardcoding for the drought variables. need to have a look at this')
        # TODO: reference ds might not be the drought mask. this functionality should be elsewhere
        spei_mask, spi_mask, ndvi_mask = create_drought_mask(output_ds[['spi','spei','ndvi']])
        ds_to_merge.append(spei_mask)
        ds_to_merge.append(spi_mask)
        ds_to_merge.append(ndvi_mask)

    output_ds = xr.merge(ds_to_merge)

    return output_ds


def check_other_netcdfs(*other_netcdfs):
    """ if dataarray check that it's named! """
    for i, xr_obj in enumerate([*other_netcdfs]):
        if isinstance(xr_obj, xr.core.dataarray.DataArray):
            assert xr_obj.name != None, f"All dataarrays must be named! Dataarray #{i+1} not named"
    return


def read_files(files):
    """ read in the files to be """
    xr_objs = []
    for file in files:
        xr_obj = xr.open_dataset(file)
        xr_objs.append(xr_obj)

    return xr_objs


# @click.command()
# @click.argument('reference_ds_path', type=click.Path(exists=True), default="EA_data/spei_spi.nc")
# @click.option('files', '--files', envvar='FILES', multiple=True, type=click.Path())
# @click.argument('output', type=click.File('wb'))
# @click.option('--drought', default=False)
if __name__ == "__main__":
    # TODO: IMPLEMENT THIS ALL IN PARALLEL

    reference_ds_path = '/soge-home/users/chri4118/EA_data/spei_spi.nc'
    # the reference ds
    reference_ds = xr.open_dataset('/soge-home/users/chri4118/EA_data/spei_spi.nc')
    reference_ds = reference_ds.rename({"value":'spi'})
    reference_ds = normalise_coordinate_names(reference_ds)

    # the output filepath
    output = "OUT.nc"

    # --------------------------------------------------------------------------
    # TODO: THIS ALL NEEDS TO BE MORE DYNAMICALLY SET UP
    # variables to join
    TEMP = xr.open_dataset("/soge-home/users/chri4118/EA_data/LST_EastAfrica.nc")
    lst_day = TEMP.lst_day
    lst_night = TEMP.lst_night
    lst_mean = (TEMP.lst_day + TEMP.lst_night) / 2
    lst_mean.name = "lst_mean"

    ET = xr.open_dataset("/soge-home/users/chri4118/EA_data/ET_EastAfrica.nc")
    evap = ET.evaporation
    baresoil_evap = ET.baresoil_evaporation
    pet = ET.potential_evaporation
    transp = ET.transpiration
    surface_sm = ET.surface_soil_moisture
    rootzone_sm = ET.rootzone_soil_moisture

    SM = xr.open_dataset('/soge-home/users/chri4118/EA_data/SM_EastAfrica.nc')
    sm = SM.sm

    PCP = xr.open_dataset('/soge-home/projects/crop_yield/chirps/EA_precip.nc')
    precip = PCP.precip

    VEG = xr.open_dataset('/soge-home/users/chri4118/EA_data/NDVI_EastAfrica.nc')
    ndvi = VEG.ndvi
    evi = VEG.evi

    # concatenate into one list to pass to the function
    vars_list = [lst_day, lst_night, lst_mean, lst_mean, evap, baresoil_evap, pet, transp, surface_sm, rootzone_sm, sm, precip, ndvi, evi]

    print(f"Reading Data from: \n{TEMP}\n{ET}\n{SM}\n{PCP}\n{VEG}")
    # --------------------------------------------------------------------------

    check_other_netcdfs(*vars_list)
    output_ds = merge_netcdfs_to_one_file(reference_ds, *vars_list, drought=True)
    save_netcdf(output_ds, output)

    print("** Process Finished **")
