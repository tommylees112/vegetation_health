from pathlib import Path
import numpy as np
from collections import namedtuple

from sklearn.metrics import mean_squared_error
from ..preprocessing import VALUE_COLS, VEGETATION_LABELS

DataTuple = namedtuple('Data', ['x', 'y', 'latlon', 'years'])


class ModelBase:

    def __init__(self, arrays=Path('data/processed/arrays'),
                 hide_vegetation=False):
        self.arrays_path = arrays
        self.hide_vegetation = hide_vegetation
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

        x = np.load(arrays_path / 'x.npy')

        if self.hide_vegetation:
            if mode == 'train':
                print('Training model without vegetation features')
            indices_to_keep = [idx for idx, val in enumerate(VALUE_COLS) if val not in VEGETATION_LABELS]

            x = x[:, :, indices_to_keep]

        return DataTuple(
                latlon=np.load(arrays_path / 'latlon.npy'),
                years=np.load(arrays_path / 'years.npy'),
                x=x,
                y=np.load(arrays_path / 'y.npy'))
