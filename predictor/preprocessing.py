import pandas as pd
import json
from pathlib import Path

KEY_COLS = ['lat', 'lon', 'time', 'gb_year', 'gb_month']
VALUE_COLS = ['lst_night', 'lst_day', 'precip', 'sm', 'spi', 'spei', 'ndvi', 'evi']
VEGETATION_LABELS = ['ndvi', 'evi']
TARGET_COL = 'ndvi'


class CSVCleaner:
    """Clean the input data, by removing nan (or encoded-as-nan) values,
    and normalizing all non-key values.
    """

    def __init__(self, raw_csv=Path('data/raw/tabular_data.csv'),
                 processed_csv=Path('data/processed/cleaned_data.csv')):

        self.csv_path = raw_csv
        self.processed_csv = processed_csv
        self.normalizing_dict = processed_csv.parents[0] / 'normalizing_dict.json'



    def readfile(self, pred_month):
        data = pd.read_csv(self.csv_path).dropna(how='any', axis=0)

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

        normalizing_dict = {}
        for col in VALUE_COLS:
            print(f'Normalizing {col}')

            series = data[col]

            # calculate normalizing values
            mean, std = series.mean(), series.std()
            # add them to the dictionary
            normalizing_dict[col] = {
                'mean': mean, 'std': std,
            }

            data[col] = (series - mean) / std

        data.to_csv(self.processed_csv, index=False)
        print(f'Saved {self.processed_csv}')

        with open(self.normalizing_dict, 'w') as f:
            json.dump(normalizing_dict, f)
        print(f'Saved {self.normalizing_dict}')

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
