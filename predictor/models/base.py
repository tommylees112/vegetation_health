from pathlib import Path
import numpy as np


class ModelBase:

    def __init__(self, arrays=Path('data/processed/arrays')):
        self.arrays_path = arrays

    def train(self):
        raise NotImplementedError

    def load_arrays(self):
        # TODO some test situation (maybe 2016)?
        latlon = np.load(self.arrays_path / 'latlon.npy')
        years = np.load(self.arrays_path / 'years.npy')
        x = np.load(self.arrays_path / 'x.npy')
        y = np.load(self.arrays_path / 'y.npy')

        return x, y, years, latlon
