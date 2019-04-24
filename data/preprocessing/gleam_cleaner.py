"""gleam_cleaner.py"""
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

import ipdb
import warnings
import os

from preprocessing.utils import (
    gdal_reproject,
    bands_to_time,
    convert_to_same_grid,
    select_same_time_slice,
    save_netcdf,
    get_holaps_mask,
    merge_data_arrays,
)

from preprocessing.cleaner import Cleaner

class GleamCleaner(Cleaner):
    def __init__(self):
        self.base_data_path = Path("/soge-home/projects/crop_yield/EGU_compare/")
        reference_data_path = self.base_data_path / "holaps_EA_clean.nc"

        # CHANGE THIS PATH:
        # ----------------------------------------------------------------------
        data_path = self.base_data_path / "EA_chirps_monthly.nc"
        # ----------------------------------------------------------------------

        # open the reference dataset
        self.reference_data_path = Path(reference_data_path)
        self.reference_ds = xr.open_dataset(self.reference_data_path).holaps_evapotranspiration

        # initialise the object using methods from the parent class
        super(ChirpsCleaner, self).__init__(data_path=data_path)

        # extract the variable of interest (TO xr.DataArray)
        self.update_clean_data(
            self.raw_data.precip, msg="Extract Precipitation from CHIRPS xr.Dataset"
        )

        # make the mask (FROM REFERENCE_DS) to copy to this dataset too
        self.get_mask()
        # self.mask = self.mask.drop('units')

    def get_mask(self):
        self.mask = get_holaps_mask(self.reference_ds)

    def convert_units(self):
        # convert unit label to 'mm day-1'
        self.clean_data.attrs["units"] = "mm day-1"


    def preprocess(self):
        # Resample the timesteps to END OF MONTH
        self.resample_time(resample_str="M")
        # select the correct time slice
        self.correct_time_slice()
        # update the units
        self.convert_units()
        # regrid to same as reference data (holaps)
        self.regrid_to_reference()
        # ipdb.set_trace()
        # use the same mask as HOLAPS
        self.use_reference_mask() # THIS GOING WRONG
        # rename data
        self.rename_xr_object("gleam_evapotranspiration")
        # save data
        save_netcdf(
            self.clean_data, filepath=self.base_data_path / "gleam_EA_clean.nc"
        )
        print("\n\n GLEAM Preprocessed \n\n")
        return
