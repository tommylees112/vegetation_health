import pandas as pd
from pathlib import Path
import xarray as xr


KEY_COLS = ['lat', 'lon', 'time', 'gb_year', 'gb_month']
VALUE_COLS = ['lst_night', 'lst_day', 'precip', 'sm', 'spi', 'spei', 'ndvi', 'evi']
VEGETATION_LABELS = ['ndvi', 'evi']
TARGET_COL = 'ndvi'


class CSVCleaner:
    """Clean the input data, by removing nan (or encoded-as-nan) values,
    and normalizing all non-key values.
    """

    def __init__(self, raw_filepath=Path('data/raw/tabular_data.csv'),
                 processed_filepath=Path('data/processed/cleaned_data.csv')):

        self.filepath = raw_filepath
        self.processed_filepath = processed_filepath

    def readfile(self, pred_month):
        data = pd.read_csv(self.filepath).dropna(how='any', axis=0)

        # a month column is already present. Add a year column
        data['time'] = pd.to_datetime(data['time'])

        data['gb_month'], data['gb_year'] = self.update_year_month(data['time'],
                                                                   pred_month)

        lst_cols = ['lst_night', 'lst_day']

        for col in lst_cols:
            # for the lst_cols, missing data is coded as 200
            data = data[data[col] != 200]

        return_cols = KEY_COLS + VALUE_COLS

        print(f'Loaded {len(data)} rows!')

        return data[return_cols]

    def process(self, pred_month=6):

        data = self.readfile(pred_month)

        data['target'] = data[TARGET_COL]

        for col in VALUE_COLS:
            print(f'Normalizing {col}')
            data[col] = self.normalize(data[col])

        data.to_csv(self.processed_filepath, index=False)
        print(f'Saved {self.processed_filepath}')

    @staticmethod
    def normalize(series):
        # all features to have 0 mean and std 1
        return (series - series.mean()) / series.std()

    @staticmethod
    def update_year_month(times, pred_month):
        """Given a pred year (e.g. 6), this method will return two new series with
        updated years and months so that a "year" of data will be the 11 months preceding the
        pred_month, and the pred_month. This makes it easier for the engineer to then make the training
        data
        """
        if pred_month == 12:
            return times.dt.month, times.dt.year

        relative_times = times - pd.DateOffset(months=pred_month)

        # we add one year so that the year column the engineer makes will be reflective
        # of the pred year, which is shifted because of the data offset we used
        return relative_times.dt.month, relative_times.dt.year + 1



class NCCleaner(CSVCleaner):
    """Clean the input data (from the .nc file), by removing nan
        (or encoded-as-nan) values, and normalising the non-key values.

        Does some preprocessing on the .nc file using xarray and then converts
         it to a dataframe for the other methods
    """


    def readfile(self, pred_month):
        # drop any Pixel-Times with missing values
        data = xr.open_dataset(self.nc_path)

        if 'month' not in [var_ for var_ in data.variables.keys()]:
            data['month'] = data['time.month']

        # a month column is already present. Add a year column
        if 'year' not in [var_ for var_ in data.variables.keys()]:
            data['year'] = data['time.year']

        data['gb_month'], data['gb_year'] = update_year_month(
            pd.to_datetime(data.time.to_series()), pred_month
        )

        # mask out the invalid temperature values
        lst_cols = ['lst_night', 'lst_day']
        for var_ in lst_cols:
            # for the lst_cols, missing data is coded as 200
            valid = (ds[var_] < 200) | (np.isnan(ds[var_]))
            ds[var_] = ds[var_].fillna(np.nan).where(valid)

        return_cols = KEY_COLS + VALUE_COLS

        # compute the ndvi_anomaly
        data['ndvi_anomaly'] = self.compute_anomaly(data.ndvi)

        # >>>>>>>>>>>> don't know how to get the dropna to work with xarray
        # convert to pd.DataFrame
        data = data.to_dataframe()
        data.dropna(how='any', axis=0)
        # <<<<<<<<<<<<<

        print(f'Loaded {len(data)} rows!')
        return data[return_cols]

    @staticmethod
    def compute_anomaly(da, time_group='time.month'):
        """ Return a dataarray where values are an anomaly from the MEAN for that
             location at a given timestep. Defaults to finding monthly anomalies.

        Arguments:
        ---------
        : da (xr.DataArray)
        : time_group (str)
            time string to group.
        """
        assert isinstance(da, xr.DataArray), f"`da` should be of type `xr.DataArray`. Currently: {type(da)}"
        mthly_vals = da.groupby(time_group).mean('time')
        da = da.groupby(time_group) - mthly_vals

        return da
