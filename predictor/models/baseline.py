import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from .base import ModelBase


class LinearModel(ModelBase):
    """A logistic regression, to be used as a baseline
    against our more complex models
    """

    def train(self):

        train_data = self.load_arrays(mode='train')

        x = train_data.x.reshape(train_data.x.shape[0], -1)

        self.model = linear_model.LinearRegression()
        self.model.fit(x, train_data.y)

        train_pred_y = self.model.predict(x)
        train_rmse = np.sqrt(mean_squared_error(train_data.y, train_pred_y))

        print(f'Train set RMSE: {train_rmse}')

    def predict(self):

        test_data = self.load_arrays(mode='test')
        x = test_data.x.reshape(test_data.x.shape[0], -1)
        test_pred_y = self.model.predict(x)
        return test_data.y, test_pred_y
