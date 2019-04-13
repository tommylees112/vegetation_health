import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_results(processed_data=Path('data/processed'), target='ndvi',
                 savefig=True):
    """Plots a landscape of the results (and optionally,
    of the ground truth)
    """

    preds = np.load(processed_data / target / 'arrays/preds.npy')
    true = np.load(processed_data / target / 'arrays/test/y.npy')
    latlon = np.load(processed_data / target / 'arrays/test/latlon.npy')

    preds_xr = pd.DataFrame(data={'preds': preds, 'lat': latlon[:, 0],
                                  'lon': latlon[:, 1]}).set_index(['lat', 'lon']).to_xarray()
    true_xr = pd.DataFrame(data={'true': true, 'lat': latlon[:, 0],
                                 'lon': latlon[:, 1]}).set_index(['lat', 'lon']).to_xarray()

    data_xr = xr.concat((preds_xr['preds'], true_xr['true']),
                        pd.Index(['predictions', 'ground truth'], name='data'))

    data_xr.plot(x='lon', y='lat', col='data', figsize=(15, 6))

    if savefig:
        plt.savefig(f'{target}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
