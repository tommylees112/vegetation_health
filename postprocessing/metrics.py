import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from collections import namedtuple
import matplotlib.pyplot as plt

# do we want to import these?
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score


def calcuate_bias(preds, true):
    """The difference between the mean of the forecasts and the mean
    of the observations. Could be expressed as a percentage of the
    mean observation. Also known as overall bias, systematic bias, or
    unconditional bias. For categorical forecasts, bias (also known
    as frequency bias) is equal to the total number of events forecast
    divided by the total number of events observed.

    $\frac{\hat{y} - y}{y} * 100$
    """
    bias = (preds.mean() - true.mean())
    pct_bias = bias / true.mean()*100

    return pct_bias
