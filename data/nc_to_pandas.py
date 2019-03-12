""" convert netcdf file to tabular dataformat (pandas)
BOILERPLATE
"""
import xarray as xr
import pandas as pd

# change this path to point to the .nc file
data_dir = ''

ds = xr.open_dataset(data_dir)

# NOTE
df = ds.to_dataframe()
df.to_csv('path/to/csv/file')
