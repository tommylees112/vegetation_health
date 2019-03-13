import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings

# invert the upside down variables
def invert(da):
    """ given a dataarray invert the latitude values so that the plots are the right way up """
    da.values = da.values[:,::-1,:]
    assert da.name, f"the dataarray passed to function invert() must have a name!"
    print(f'inverted the values of {da.name}')
    return da


def get_lc_mask(ds, mask_var):
    assert mask_var in ["spi", "spei", "ndvi"], f"Mask Variable should be one of spi spei ndvi (ADD MORE)"
    # create a land cover mask
    warnings.warn('Currently get_lc_mask() is working with hardcoded SPI values. May cause problems')
    lc_mask = ~ds['spi'].isel(time=1).isnull()
    lc_mask.name = "lc_mask"
    print("created lc_mask")
    return lc_mask


def remove_excess_parameters(ds):
    """ select only some of the variables (NOTE: poorly programmed because hardcoded)
         TODO: remove hardcoded params
    """
    key_parameters = ["lst_mean","lst_day","lst_night","sm","precip","evaporation","transpiration","spi","spei","ndvi","evi","surface_soil_moisture","rootzone_soil_moisture","baresoil_evaporation","potential_evaporation"]
    # check that all the parameters exist in the dataset
    for param in key_parameters:
        assert param in [var for var in ds.variables.keys()], f"Param {param} not found!"
    ds = ds[key_parameters]
    print(f"Keeping parameters: {key_parameters}")
    return ds


def mask_sea(ds, lc_mask):
    """ MASK THE SEA VALUES (select only where lc_mask == 1)"""
    ds = ds.where(lc_mask)
    print(f"Sea values masked for ds with vars {[var for var in ds.data_vars.keys()]}")
    return ds


def get_boolean_drought_ds(ds):
    """ extract the drought indices from the data array """
    assert "drought_ndvi" in [var for var in ds.variables.keys()], f"drought_ndvi should be in {[var for var in ds.variables.keys()]}"
    assert "drought_spei" in [var for var in ds.variables.keys()], f"drought_spei should be in {[var for var in ds.variables.keys()]}"
    assert "drought_spi" in [var for var in ds.variables.keys()], f"drought_spi should be in {[var for var in ds.variables.keys()]}"
    drought = ds[['drought_spei','drought_spi','drought_ndvi']]
    print("Created a drought index xr.Dataset")
    return drought


def save_netcdf(output_ds, filename):
    """ save the dataset"""
    output_ds.to_netcdf(filename)
    print(f"{filename} saved!")
    return


def clean_lst_variables(ds):
    """"""
    lst_vars = [var_ for var_ in ds.data_vars.keys() if "lst" in var_]
    for lst_var in lst_vars:
        # filter OUT the lst values of 200
        valid = ds[lst_var] < 200
        ds[lst_var] = ds[lst_var].fillna(np.nan).where(valid)
    return ds


def clean_data(ds, lc_mask):
    # drop the final timestep
    ds = ds.isel(time=slice(0,-1))

    # invert precip
    ds['precip'] = invert(ds.precip)

    # get only the important parameters
    ds = remove_excess_parameters(ds)

    # mask out the sea
    ds = mask_sea(ds, lc_mask)

    # clean lst variables
    ds = clean_lst_variables(ds)

    return ds


def get_df_of_pixels_to_remove(lc_mask):
    """ return a dataframe of the pixels that correspond to SEA points (and therefore that we want to remove)
    """
    mask_df = lc_mask.to_dataframe()
    mask_df = mask_df.reset_index().drop(columns=["time"])
    indexes_to_remove = mask_df.where(~mask_df.lc_mask).dropna()

    return indexes_to_remove



def shift_by_time(ds):
    """ shift dataset by number of timesteps (so if shift by 3 you get )"""
    return ds.shift(ts)



def mask_ds(ds, drought_mask, drought=True):
    """set all of the pixels that are SEA or DROUGHT to NAN"""
    print("drought pixels masked")

    return ds


def make_masks_boolean(mask_ds):
    """ convert masks from 0,1 to False,True """
    try:
        print(f"Convert mask to bool for ds with vars {[var for var in mask_ds.data_vars.keys()]}")
        mask_ds = mask_ds.astype(bool)
    except:
        try:
            print(f"UNABLE to convert mask to bool for ds with vars {[var for var in mask_ds.data_vars.keys()]}")
        except: # is a data array and it doesn't have .data_vars()
            print(f"UNABLE to convert mask to bool for ds with vars {mask_ds.name}")
    return mask_ds


def read_data(data_path='.', mask_var='spi'):
    """ """
    assert os.path.isfile(data_path), f"The path provided to read data does not exist! Currently: {data_path}"
    # data_path = "/Volumes/Lees_Extend/data/ea_data/all_variables_LC2.nc"
    print(f"Reading from file: {data_path}")
    ds = xr.open_dataset(data_path)
    lc_mask = get_lc_mask(ds, mask_var)
    # --------------------------------------------------------------------------
    # OFFENDING LINE
    warnings.warn('drought_ndvi hardcoded in here. Not at all okay.gst FIX ME')
    ds['drought_ndvi'] = ds.ndvi < (ds.ndvi.mean(dim='time') - ds.ndvi.std(dim='time'))
    # --------------------------------------------------------------------------
    drought_mask = get_boolean_drought_ds(ds)

    # REMOVE THE SEA VALUES from drought mask
    drought_mask = mask_sea(drought_mask, lc_mask)
    drought_mask = make_masks_boolean(drought_mask)
    lc_mask = make_masks_boolean(lc_mask)

    ds = clean_data(ds, lc_mask)

    return ds, lc_mask, drought_mask


def print_shift_explanation(ds):
    """ print statements to explain the differences with the shift operator """
    print("*** UNSHIFTED TIME ***")
    print("time=0\n", ds.isel(lat=slice(0,2), lon=slice(0,2),time=0).precip.values)
    print("time=1\n", ds.isel(lat=slice(0,2), lon=slice(0,2),time=1).precip.values)
    print("time=2\n",ds.isel(lat=slice(0,2), lon=slice(0,2), time=2).precip.values)
    print("time=3\n",ds.isel(lat=slice(0,2), lon=slice(0,2), time=3).precip.values)
    print()
    print("*** SHIFTED TIME (+1) = moving the HISTORICAL TIMESTEPS FORWARD to the PRESENT ***")
    print("time=0\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=1).isel(time=0).precip.values)
    print("time=1\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=1).isel(time=1).precip.values)
    print("time=2\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=1).isel(time=2).precip.values)
    print("time=3\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=1).isel(time=3).precip.values)
    print()
    print("*** SHIFTED TIME (-1) = moving the PRESENT TIMESTEPS BACKWARD to the PAST ***")
    print("time=0\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=-1).isel(time=0).precip.values)
    print("time=1\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=-1).isel(time=1).precip.values)
    print("time=2\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=-1).isel(time=2).precip.values)
    print("time=3\n",ds.isel(lat=slice(0,2), lon=slice(0,2)).shift(time=-1).isel(time=3).precip.values)
    print()

# ----------------------------------------------------------------------
def create_lc_mask(ds):
    """ from the dataset create a landcover mask """
    # create a land cover mask
    lc_mask = ~ds.spi.isel(time=1).isnull()
    lc_mask.name = "lc_mask"

    # create df lc mask
    mask_df = lc_mask.to_dataframe()
    mask_df = mask_df.reset_index().drop(columns=["time"])

    # get df of SEA pixels (pixels to remove)
    indexes_to_remove = mask_df.where(~mask_df.lc_mask).dropna()

    return lc_mask, mask_df, indexes_to_remove
