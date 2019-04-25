import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
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


def plot_results(processed_data=Path('data/processed'), target='ndvi',
                 plot_difference=False, savefig=True):
    """Plots a landscape of the results (and optionally,
    of the ground truth)
    """

    preds = np.load(processed_data / target / 'arrays/preds.npy')
    true = np.load(processed_data / target / 'arrays/test/y.npy')
    latlon = np.load(processed_data / target / 'arrays/test/latlon.npy')

    preds_xr = create_dataset_from_vars(preds, latlon, "preds", to_xarray=True)
    true_xr = create_dataset_from_vars(true, latlon, "true", to_xarray=True)

    data_xr = xr.concat((preds_xr['preds'], true_xr['true']),
                        pd.Index(['predictions', 'ground truth'], name='data'))

    if plot_difference:
        # compute the difference and create a difference plot
        data = data_xr.data[1] - data_xr.data[0]
        da = xr.DataArray(data, coords=[data_xr.lat, data_xr.lon], dims=['lat','lon'])
        data_xr = da.to_dataset('difference')

        data_xr.difference.plot(x='lon', y='lat', figsize=(15, 6))
    else:
        data_xr.plot(x='lon', y='lat', col='data', figsize=(15, 6))

    if savefig:
        plt.savefig(f'{target}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
