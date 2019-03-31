from pathlib import Path
import pandas as pd
import numpy as np

from .preprocessing import VALUE_COLS


class Engineer:
    """Take the clean csv file and turn it into numpy arrays ready
    for training
    """

    def __init__(self, cleaned_data=Path('data/processed/cleaned_data.csv'),
                 arrays=Path('data/processed/arrays')):

        self.arrays_path = arrays
        if not self.arrays_path.exists():
            self.arrays_path.mkdir()
        self.cleaned_data_path = cleaned_data

    def readfile(self):
        return pd.read_csv(self.cleaned_data_path)

    def process(self, test_year=2016):

        data = self.readfile()

        # outputs
        latlons, years, vals, targets = [], [], [], []

        skipped = 0
        # first, groupby lat, lon, so that we process the same place together
        for latlon, group in data.groupby(by=['lat', 'lon']):
            latlon_np = np.array(latlon)
            for year, subgroup in group.groupby(by='gb_year'):
                if len(subgroup) != 12:
                    # print(f'Ignoring data from {year} at {latlon} due to missing rows')
                    skipped += 1
                    continue
                subgroup = subgroup.sort_values(by='gb_month', ascending=True)

                x = subgroup[:-1][VALUE_COLS].values
                y = subgroup.iloc[-1]['target']

                latlons.append(latlon_np)
                years.append(year)
                vals.append(x)
                targets.append(y)

                if len(latlons) % 1000 == 0:
                    print(f'Processed {len(latlons)} examples')

        print(f'Done processing {len(latlons)} years! Skipped {skipped} years due to missing rows')

        # turn everything into np arrays for manipulation
        latlons, years, vals, targets = np.vstack(latlons), np.array(years), np.stack(vals), np.array(targets)

        # split into train and test sets
        test_idx = np.where(years == test_year)[0]
        train_idx = np.where(years < test_year)[0]

        test_arrays = self.arrays_path / 'test'
        train_arrays = self.arrays_path / 'train'

        test_arrays.mkdir(parents=True, exist_ok=True)
        train_arrays.mkdir(exist_ok=True)

        print('Saving data')
        np.save(train_arrays / 'latlon.npy', latlons[train_idx])
        np.save(train_arrays / 'years.npy', years[train_idx])
        np.save(train_arrays / 'x.npy', vals[train_idx])
        np.save(train_arrays / 'y.npy', targets[train_idx])

        np.save(test_arrays / 'latlon.npy', latlons[test_idx])
        np.save(test_arrays / 'years.npy', years[test_idx])
        np.save(test_arrays / 'x.npy', vals[test_idx])
        np.save(test_arrays / 'y.npy', targets[test_idx])
