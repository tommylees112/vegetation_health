import pandas as pd
import numpy as np
from pathlib import Path

KEY_COLS = ['lat', 'lon', 'month', 'year']
VALUE_COLS = ['lst_night', 'lst_day', 'precip', 'sm', 'spi', 'spei', 'ndvi', 'evi']


class CSVCleaner:
    """Clean the input data, by removing nan (or encoded-as-nan) values,
    and normalizing all non-key values.
    """

    def __init__(self, raw_csv=Path('data/raw/tabular_data.csv'),
                 processed_csv=Path('data/processed/cleaned_data.csv')):

        self.csv_path = raw_csv
        self.processed_csv = processed_csv

    def readfile(self):
        data = pd.read_csv(self.csv_path).dropna(how='any', axis=0)

        # a month column is already present. Add a year column
        data['year'] = pd.to_datetime(data['time']).dt.year

        lst_cols = ['lst_night', 'lst_day']

        for col in lst_cols:
            # for the lst_cols, missing data is coded as 200
            data = data[data[col] != 200]

        return_cols = KEY_COLS + VALUE_COLS

        print(f'Loaded {len(data)} rows!')

        return data[return_cols]

    def process(self, normalizing_percentile=95):

        data = self.readfile()

        for col in VALUE_COLS:
            print(f'Normalizing {col}')
            data[col] = self.normalize(data[col], normalizing_percentile)

        data.to_csv(self.processed_csv, index=False)
        print(f'Saved {self.processed_csv}')

    @staticmethod
    def normalize(series, normalizing_percentile):
        # all features to have 0 mean and a normalized point to point value
        min_percentile = (100 - normalizing_percentile) / 2
        max_percentile = 100 - min_percentile
        ptp = np.percentile(series, max_percentile) - np.percentile(series, min_percentile)

        return (series - series.mean()) / ptp
