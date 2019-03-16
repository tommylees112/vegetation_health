from pathlib import Path
import numpy as np
from collections import namedtuple

from sklearn.metrics import mean_squared_error


class ModelBase:

    def __init__(self, arrays=Path('data/processed/arrays')):
        self.arrays_path = arrays
        self.model = None  # to be added by the model classes

    def train(self):
        raise NotImplementedError

    def predict(self):
        # This method should return the predictions, and
        # the corresponding true values, read from the test
        # arrays
        raise NotImplementedError

    def evaluate(self, return_eval=False):
        y_true, y_pred = self.predict()

        test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f'Test set RMSE: {test_rmse}')

        if return_eval:
            return test_rmse

    def load_arrays(self, mode='train'):

        arrays_path = self.arrays_path / mode

        Data = namedtuple('Data', ['x', 'y', 'latlon', 'years'])

        return Data(
                latlon=np.load(arrays_path / 'latlon.npy'),
                years=np.load(arrays_path / 'years.npy'),
                x=np.load(arrays_path / 'x.npy'),
                y=np.load(arrays_path / 'y.npy'))
