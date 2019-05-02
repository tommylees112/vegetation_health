import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from collections import namedtuple
import matplotlib.pyplot as plt


def create_dataset_from_vars(vars, latlon, varname, to_xarray=True):
    """ Convert the variables from `np.array` to `pd.DataFrame`
    and optionally `xr.Dataset`. By default converts to `xr.Dataset`

    Arguments:
    ---------
    : vars (np.array)
        the values of the variable of interest (e.g. Predictions of NDVI from model)

    : latlon (np.array)
        the latlon location for each of the values in vars

    : varname (str)
        the name of the variable

    TODO:
    ----
    * Implement a method that works with TIME so that the xarray objects
        have a time dimension too
    """
    assert len(vars) == len(latlon), f"The length of the latlons array should be the same as the legnth of the vars array. Currently latlons: {len(latlon)} vars: {len(vars)}"


    df = pd.DataFrame(data={varname: vars, 'lat': latlon[:, 0],
                      'lon': latlon[:, 1]}).set_index(['lat', 'lon'])
    if to_xarray:
        return df.to_xarray()
    else:
        return df


# ------------------------------------------------------------------------------
# Engineering Utilities
# ------------------------------------------------------------------------------

def drop_nans_and_flatten(dataArray):
    """flatten the array and drop nans from that array. Useful for plotting histograms.

    Arguments:
    ---------
    : dataArray (xr.DataArray)
        the DataArray of your value you want to flatten
    """
    # drop NaNs and flatten
    return dataArray.values[~np.isnan(dataArray.values)]

# ------------------------------------------------------------------------------
# Working with Time Variables
# ------------------------------------------------------------------------------


def compute_anomaly(da, time_group='time.month'):
    """ Return a dataarray where values are an anomaly from the MEAN for that
         location at a given timestep. Defaults to finding monthly anomalies.

    Arguments:
    ---------
    : da (xr.DataArray)
    : time_group (str)
        time string to group.
    """
    mthly_vals = da.groupby(time_group).mean('time')
    da = da.groupby(time_group) - mthly_vals

    return da


# ------------------------------------------------------------------------------
# Collapsing Time Dimensions
# ------------------------------------------------------------------------------


def caclulate_std_of_mthly_seasonality(ds,double_year=False):
    """Calculate standard deviataion of monthly variability """
    std_ds = calculate_monthly_std(ds)
    seasonality_std = calculate_spatial_mean(std_ds)

    # rename vars
    var_names = get_non_coord_variables(seasonality_std)
    new_var_names = [var + "_std" for var in var_names]
    seasonality_std = seasonality_std.rename(dict(zip(var_names, new_var_names)))

    #
    if double_year:
        seasonality_std = create_double_year(seasonality_std)

    return seasonality_std


def calculate_monthly_mean(ds):
    assert 'time' in [dim for dim in ds.dims.keys()], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby('time.month').mean(dim='time')


def calculate_monthly_std(ds):
    assert 'time' in [dim for dim in ds.dims.keys()], f"Time must be in the dataset dimensions. Currently: {[dim for dim in ds.dims.keys()]}"
    return ds.groupby('time.month').std(dim='time')


def calculate_monthly_mean_std(ds):
    """ """
    # calculate mean and std
    mean = calculate_monthly_mean(ds)
    std = calculate_monthly_std(ds)

    # get var names
    dims = [dim for dim in mean.dims.keys()]
    vars = [var for var in mean.variables.keys() if var not in dims]

    # rename vars so can return ONE ds
    mean_vars = [var+'_monmean' for var in vars]
    std_vars = [var+'_monstd' for var in vars]
    mean = mean.rename(dict(zip(vars, mean_vars)))
    std = std.rename(dict(zip(vars, std_vars)))

    return xr.merge([mean, std])


def calculate_spatial_mean(ds):
    assert ('lat' in [dim for dim in ds.dims.keys()]) & ('lon' in [dim for dim in ds.dims.keys()]), f"Must have 'lat' 'lon' in the dataset dimensisons"
    return ds.mean(dim=['lat','lon'])


def create_double_year(seasonality):
    """for seasonality data (values for each month) return a copy for a second
        year to account for the cross-over between DJF

    Returns:
    -------
    : (xr.Dataset)
        a Dataset object with 24 months (2 annual cycles)
    """
    assert 'month' in [coord for coord in seasonality.coords.keys()], f"`month` must be a present coordinate in the seasonality data passed to the `create_double_year` function! Currently: {[coord for coord in seasonality.coords.keys()]}"

    seas2 = seasonality.copy()
    seas2['month'] = np.arange(13,25)

    # merge the 2 datasets
    return xr.merge([seasonality, seas2])



# ------------------------------------------------------------------------------
# Adding Features to xarray
# ------------------------------------------------------------------------------

def create_double_year(seasonality):
    """for seasonality data (values for each month) return a copy for a second
        year to account for the cross-over between DJF

    Returns:
    -------
    : (xr.Dataset)
        a Dataset object with 24 months (2 annual cycles)
    """
    assert 'month' in [coord for coord in seasonality.coords.keys()], f"`month` must be a present coordinate in the seasonality data passed to the `create_double_year` function! Currently: {[coord for coord in seasonality.coords.keys()]}"

    seas2 = seasonality.copy()
    seas2['month'] = np.arange(13,25)

    # merge the 2 datasets
    return xr.merge([seasonality, seas2])


def replace_with_dict(ar, dic):
    """ Replace the values in an np.ndarray with a dictionary

    https://stackoverflow.com/a/47171600/9940782

    """
    assert isinstance(ar, np.ndarray), f"`ar` shoule be a numpy array! (np.ndarray). To work with xarray objects, first select the values and pass THESE to the `replace_with_dict` function (ar = da.values) \n Type of `ar` currently: {type(ar)}"
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    # NOTE: something going wrong with the number for the indices (0 based vs. 1 based)
    warnings.warn('We are taking one from the index. need to check this is true!!!')
    return v[sidx[ np.searchsorted(k, ar, sorter=sidx) -1 ] ]



def replace_with_dict2(ar, dic):
    """Replace the values in an np.ndarray with a dictionary

    https://stackoverflow.com/a/47171600/9940782
    """
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    warnings.warn('We are taking one from the index. need to check this is true!!!')
    return vs[np.searchsorted(ks,ar) -1 ]


# TODO: rename this function
def get_lookup_val(xr_obj, variable, new_variable, lookup_dict):
    """ Assign a new Variable to xr_obj with values from lookup_dict.
    Arguments:
    ---------
    : xr_obj (xr.Dataset, xr.DataArray)
        the xarray object we want to look values up from
    : variable (str)
        the INPUT variable we are hoping to look the values up from (the dictionary keys)
    : new_variable (str)
        the name of the OUTPUT variable we want to put the dictionary values in
    : lookup_dict (dict)
        the dictionary we want to lookup the values of 'variable' in to return values to 'new_variable'
    """
    # get the values as a numpy array
    if isinstance(xr_obj, xr.Dataset):
        ar = xr_obj[variable].values
    elif isinstance(xr_obj, xr.DataArray):
        ar = xr_obj.values
    else:
        assert False, f"This function only works with xarray objects. Currently xr_obj is type: {type(xr_obj)}"

    assert isinstance(ar, np.ndarray), f"ar should be a numpy array!"
    assert isinstance(lookup_dict, dict), f"lookup_dict should be a dictionary object!"

    # replace values in a numpy array with the values from the lookup_dict
    new_ar = replace_with_dict2(ar, lookup_dict)

    # assign the values looked up from the dictionary to a new variable in the xr_obj
    new_da = xr.DataArray(new_ar, coords=[xr_obj.lat, xr_obj.lon], dims=['lat', 'lon'])
    new_da.name = new_variable
    xr_obj = xr.merge([xr_obj, new_da])

    return xr_obj


# ------------------------------------------------------------------------------
# Subsetting Utilities
# ------------------------------------------------------------------------------


def subset_of_predns(preds, true, condition):
    """Select subset of arrays (predictions and true) based
    on a condition (boolean array)

    Example:
    -------
    >>> processed_data=Path('data/processed')
    >>> true = np.load(processed_data / "ndvi" / 'arrays/test/y.npy')
    >>> low_ndvi_thresh = np.quantile(true, 0.2)
    >>> condition_low = true < low_ndvi_thresh
    >>> pred_low,true_low = subset_of_predns(preds, true, condition_low)
    """
    assert (len(preds) == len(true))&(len(condition) == len(true)), f"The arrays must be the same length! Currently:\npreds: {len(preds)} true: {len(true)} condition: {len(condition)}"
    return preds[condition], true[condition]


def get_mask_based_on_error_condition(ds, q, gt_lt="gt"):
    """create a mask based on the error being `<` or `>` q=q

    Example:
    -------
    # create as mask based on difference being >q=0.95 (TOP 5% of errors)
    >>> mask = get_mask_based_on_error_condition(ds, q=0.95, gt_lt="gt")
    """
    coords=[coord for coord in ds.coords.keys()]
    vars_=[var for var in ds.variables.keys() if var not in coords]
    assert "difference" in vars_, f"Must have a difference variable in the ds object"

    if gt_lt == "gt":
        cond = abs(ds.difference) > abs(ds.difference).quantile(q=q)
    else:
        cond = abs(ds.difference) < abs(ds.difference).quantile(q=q)

    mask = ~((ds.where(cond)))[vars_[0]].isnull()
    mask = mask.drop('quantile')
    return mask
