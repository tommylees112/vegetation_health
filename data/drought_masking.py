"""
@tommylees112

Code for getting land surface variables for previous timesteps that are IN
 drought or OUT of drought.

For the pixels that are in drought, return variables for that pixel for t=0,
 t=-1,t=-2,t=-3.

This way you have the previous 3 months worth of data
"""
import xarray as xr
from joblib import Parallel, delayed
import pandas as pd
import os
from utils import shift_by_time, read_data, print_shift_explanation
from utils import save_netcdf
import warnings
import numpy as np


def get_monthly_location_specific_drought_mask(ds, variable):
    """ from a dataset, use a variable and calculate 1 STD below
         the mean conditioned on:
         a) LOCATION (lat/lon)
         b) TIME (month)
    """
    # get month labels
    ds = ds.assign_coords(**{'month':ds.time.dt.month})

    outs = []
    for month in range(0,13):
        # select ONE MONTH for your variable of interest from the dataset
        d = ds[variable].sel(time=(ds.time.dt.month == month))

        # calculate avg/std
        avg = ds[variable].isel(time=(ds.time.dt.month == month)).mean('time')
        std_dev = ds[variable].isel(time=(ds.time.dt.month == month)).std('time')

        # calculate threshold
        threshold = avg - std_dev

        # calculate mask
        mask = d < threshold

        # create month specific mask for each timestep
        outs.append(mask)
        drought_ndvi = xr.concat(outs, dim="time")
    return drought_ndvi



def create_drought_mask(drought_ds):
    """ at the moment just use an SPI / SPEI of <= -1
        TODO: implement more flexible functionality
    """
    assert ("spi" in [_ for _ in drought_ds.var().variables])&("spei" in [_ for _ in drought_ds.var().variables]), f"spi and spei are expected to be in the drought_ds netcdf file. Currently {drought_ds.var().variables}"
    assert "ndvi" in [_ for _ in drought_ds.var().variables], f"ndvi should be a variable. currently: {drought_ds.var().variables}"

    # turn the values less than -1 SD into mask (SAME DIMS AS RAW DS)
    warnings.warn('NOTE: here the code could do with being more flexible in the determination of drought thresholds and then applying them.')
    warnings.warn('Use Ellens functionality already built in Akkadia')
    spei_drought = drought_ds.spei.where(drought_ds.spei < -1)
    spi_drought = drought_ds.spi.where(drought_ds.spi < -1)

    ndvi_drought = get_monthly_location_specific_drought_mask(drought_ds, 'ndvi')

    # turn into a boolean mask
    drought_spei = spei_drought.notnull()
    drought_spei = drought_spei.rename('drought_spei')
    drought_spi = spi_drought.notnull()
    drought_spi = drought_spi.rename('drought_spi')
    drought_ndvi = ndvi_drought.notnull()
    drought_ndvi = drought_ndvi.rename('drought_ndvi')

    return drought_spei, drought_spi, drought_ndvi


def mask_drought_events(ds, drought_mask, ts, index, drought=True):
    """ for a given drought mask, at a given time, return the previous 3 months of data
         for pixels that were IN (or out of) drought
    """
    assert ts > 3, "Need timestep to be greater than 3 because want the PREVIOUS 3 months. Otherwise all nans!"
    assert index in ['spei','spi', 'ndvi'], f"index must be either ['spei', 'spi', 'ndvi']. Currently: {index}"

    # get the drought mask at that timestep (NOTE have to invert BEFORE this)
    if index=='spi':
        msk = drought_mask.drought_spi.isel(time=ts)
    elif index=='spei':
        msk = drought_mask.drought_spei.isel(time=ts)
    elif index=='ndvi':
        msk = drought_mask.drought_ndvi.isel(time=ts)

    # if want pixels that were IN DROUGHT
    if drought:
        t0 = ds.where(msk).isel(time=ts)
        t1 = ds.where(msk).isel(time=ts-1)
        t2 = ds.where(msk).isel(time=ts-2)
        t3 = ds.where(msk).isel(time=ts-3)

    # if want pixels that are NOT IN DROUGHT
    else:
        t0 = ds.where(msk).isel(time=ts)
        t1 = ds.where(msk).isel(time=ts-1)
        t2 = ds.where(msk).isel(time=ts-2)
        t3 = ds.where(msk).isel(time=ts-3)

    return [t0, t1, t2, t3]


def merge_all_ts_into_one_ds(ds_arr):
    """merge all of the timesteps (t=0, t-1, t-2, t-3) into one dataset object

    input:
    : array of all the
    returns:
    : ds_out (xr.Dataset): one dataset with all of the variables
    """
    # assert that the minimum time is the first dataset in the array
    time = ds_arr[0].time
    ds_rnm = []

    for ts, ds_ in enumerate(ds_arr):
        # create dict of variables to rename
        map_names = dict(zip([var for var in ds_.variables.keys() if var not in ['time','lat','lon']],
                     [f"{var}_t{ts}" for var in ds_.variables.keys() if var not in ['time','lat','lon']])
                    )
        # rename the variables
        ds_ = ds_.rename(map_names)
        ds_rnm.append(ds_)

    # drop the 'time' (TO ALLOW THE MERGE)
    ds_rnm = [ds_.drop('time') for ds_ in ds_rnm]

    # merge the variables
    ds_out = xr.merge(ds_rnm)

    # reassign 'Time' from the first dataset
    ds_out = ds_out.assign_coords(**{'time':time})

    return ds_out


def calculate_drought_masked_ds(ds, drought_mask, ts):
    """ run the above functions on EVERY timestep
         so output a dataset with each timestep a calculation of the previous months
    """
    print(f"Extracting Drought vars from timestep {ts}")
    ds_arr = mask_drought_events(ds, drought_mask, ts=ts, index='ndvi', drought=True)
    ds_drought = merge_all_ts_into_one_ds(ds_arr)

    return ds_drought


def calculate_Ndrought_masked_ds(ds, drought_mask, ts):
    """ run the above functions on EVERY timestep
         so output a dataset with each timestep a calculation of the previous months
    """
    print(f"Extracting NON-Drought vars from timestep {ts}")
    ds_arr = mask_drought_events(ds, drought_mask, ts=ts, index='ndvi', drought=False)
    ds_Ndrought = merge_all_ts_into_one_ds(ds_arr)

    return ds_Ndrought


def drought_across_all_timesteps(ds, drought_mask, start_ts=4):
    """ run the drought masking process for ALL TIMESTEPS

    Note: the output of Parallel(n_jobs=2)({FNCTN}) is a list of all of the variables
           because they are timestamped it doesn't matter if they come back in a different
           order. Therefore, we save the reordering to a later date. Future me can deal
           with that problem.
    """
    dr = []
    Ndr = []

    # RUN the drought first
    print("Extracting the pixels in DROUGHT")
    with Parallel(n_jobs=30, verbose=True) as parallel:
        dr = parallel(
                delayed(calculate_drought_masked_ds)
                (ds=ds,drought_mask=drought_mask,ts=ts) for ts in range(start_ts, ds.time.shape[0])
                )
    print("Pixels in DROUGHT extracted")

    # then run the Ndrought
    print("Extracting the pixels NOT in DROUGHT")
    # convert to boolean arrays in order to INVERT them
    Ndrought_mask = ~(drought_mask.astype(bool))
    with Parallel(n_jobs=30, verbose=True) as parallel:
        Ndr = parallel(
                delayed(calculate_Ndrought_masked_ds)
                (ds=ds,drought_mask=Ndrought_mask,ts=ts) for ts in range(start_ts, ds.time.shape[0])
                )
    print("Pixels NOT in DROUGHT extracted")

    # concatenate the arrays by time into ONE dataset (agree there's lots of duplication of data here)
    # TODO: does this really make sense? we are duplicating SOOOOOOO much data and for what?
    #       definitely a better way.
    ds_drought = xr.concat(dr, dim='time')
    ds_drought.to_netcdf('ds_drought.nc')
    print("DROUGHT Variables extracted. Saving to netcdf ds_drought.nc ...")

    ds_Ndrought = xr.concat(Ndr, dim='time')
    ds_Ndrought.to_netcdf('ds_Ndrought.nc')
    print("NOT DROUGHT Variables extracted. Saving to netcdf ds_Ndrought.nc ...")

    return ds_drought, ds_Ndrought


def run_drought_processing(data_path, mask_var='spi'):
    """

    """
    print("** Running drought processing for all timesteps! **")
    ds, lc_mask, drought_mask = read_data(data_path, mask_var)
    print(f"** Data read in - using {mask_var} as the masking variable **")
    ds_drought, ds_Ndrought = drought_across_all_timesteps(ds, drought_mask)

    save_netcdf(ds_drought, "ds_drought.nc")
    save_netcdf(ds_Ndrought, "ds_Ndrought.nc")
    print("** Process finished **")

    return ds_drought, ds_Ndrought





def fix_drought_var():
    """if drought variable not in dataset then append it!"""
    pass


if __name__=="__main__":
    """ TODO: set up as a CLI """
    # data_path = "/soge-home/users/chri4118/EA_data/all_variables_LCMASK.nc"
    data_path = "/soge-home/users/chri4118/ea_exploration/OUT.nc"
    ds_drought, ds_Ndrought = run_drought_processing(data_path, mask_var='ndvi')
