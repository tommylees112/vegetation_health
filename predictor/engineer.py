from pathlib import Path
import pandas as pd
import numpy as np

from .preprocessing import VALUE_COLS, TARGET_COL


class Engineer:
    """Take the clean csv file and turn it into numpy arrays ready
    for training
    """

    def __init__(self, cleaned_data=Path('data/processed/cleaned_data.csv'),
                 arrays=Path('data/processed/arrays')):

        if not arrays.exists():
            arrays.mkdir()
        self.arrays_path = arrays
        self.cleaned_data_path = cleaned_data

    def readfile(self):
        return pd.read_csv(self.cleaned_data_path)

    def process(self):

        data = self.readfile()

        # outputs
        latlons, years, vals, targets = [], [], [], []

        skipped = 0
        # first, groupby lat, lon, so that we process the same place together
        for latlon, group in data.groupby(by=['lat', 'lon']):
            latlon_np = np.array(latlon)
            for year, subgroup in group.groupby(by='year'):
                if len(subgroup) != 12:
                    # print(f'Ignoring data from {year} at {latlon} due to missing rows')
                    skipped += 1
                    continue
                subgroup = subgroup.sort_values(by='month', ascending=True)

                x = subgroup[:-1][VALUE_COLS].values
                y = subgroup.iloc[-1]['target']

                latlons.append(latlon_np)
                years.append(year)
                vals.append(x)
                targets.append(y)

                if len(latlons) % 1000 == 0:
                    print(f'Processed {len(latlons)} examples')

        print(f'Done processing {len(latlons)} years! Skipped {skipped} years due to missing rows')

        print('Saving data')
        np.save(self.arrays_path / 'latlon.npy', np.vstack(latlons))
        np.save(self.arrays_path / 'years.npy', np.array(years))
        np.save(self.arrays_path / 'x.npy', np.stack(vals))
        np.save(self.arrays_path / 'y.npy', np.array(targets))
