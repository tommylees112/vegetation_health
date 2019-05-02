import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt




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
        ds = da.to_dataset('difference')

        ds.difference.plot(x='lon', y='lat', figsize=(12, 8))
    else:
        data_xr.plot(x='lon', y='lat', col='data', figsize=(15, 6))

    if savefig:
        plt.savefig(f'{target}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
