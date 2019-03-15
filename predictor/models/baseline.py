import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .base import ModelBase


class LinearModel(ModelBase):

    def train(self):

        x, y, _, _ = self.load_arrays()

        x = x.reshape(x.shape[0], -1)

        train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                            test_size=0.3,
                                                            shuffle=True)

        model = linear_model.LinearRegression()
        model.fit(train_x, train_y)

        train_pred_y = model.predict(train_x)
        train_rmse = np.sqrt(mean_squared_error(train_y, train_pred_y))

        print(f'Train set RMSE: {train_rmse}')

        # test
        test_pred_y = model.predict(test_x)
        test_rmse = np.sqrt(mean_squared_error(test_y, test_pred_y))

        print(f'Test set RMSE: {test_rmse}')
