from pathlib import Path
import xarray as xr
import numpy as np

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


# ------------------------------------------------------------------------------
# HOLAPS cleaner
# ------------------------------------------------------------------------------


class SpeiCleaner(Cleaner):
    """Preprocess the HOLAPS dataset"""
    assert False, "This is just boilerplate code from another"
    def __init__(self):
        # init data paths (should be arguments)
        self.base_data_path = Path("/soge-home/projects/crop_yield/EGU_compare/")
        data_path = self.base_data_path / "holaps_africa.nc"
        reproject_path = self.base_data_path / "holaps_africa_reproject.nc"

        super(HolapsCleaner, self).__init__(data_path=data_path)
        self.reproject_path = Path(reproject_path)


    def chop_EA_region(self, outfile_path):
        """ cheeky little bit of bash scripting with string interpolation (kids don't try this at home) """
        in_file = self.base_data_path / "holaps_reprojected.nc"
        out_file = self.base_data_path / "holaps_EA.nc"
        lonmin = 32.6
        lonmax = 51.8
        latmin = -5.0
        latmax = 15.2

        cmd = (
            f"cdo sellonlatbox,{lonmin},{lonmax},{latmin},{latmax} {in_file} {out_file}"
        )
        print(f"Running command: {cmd}")
        os.system(cmd)
        print("Chopped East Africa from the Reprojected data")
        re_chopped_data = xr.open_dataset(out_file)
        self.update_clean_data(
            re_chopped_data, msg="Opened the reprojected & chopped data"
        )
        return


    def reproject(self):
        """ reproject to WGS84 / geographic latlon """
        if not self.reproject_path.is_file():
            gdal_reproject(infile=self.data_path, outfile=self.reproject_path)

        repr_data = xr.open_dataset(self.reproject_path)

        # get the timestamps from the original holaps data
        h_times = self.clean_data.time
        # each BAND is a time (multiple raster images 1 per time)
        repr_data = bands_to_time(repr_data, h_times, var_name="LE_Mean")

        # TODO: ASSUMPTION / PROBLEM
        warnings.warn(
            "TODO: No idea why but the values appear to be 10* bigger than the pre-reprojected holaps data"
        )
        repr_data /= 10  # WHY ARE THE VALUES 10* bigger?

        self.update_clean_data(repr_data, "Data Reprojected to WGS84")

        save_netcdf(
            self.clean_data, filepath=self.base_data_path / "holaps_reprojected.nc"
        )
        return


    def convert_units(self):
        # Convert from latent heat (w m-2) to evaporation (mm day-1)
        holaps_mm = self.clean_data / 28
        holaps_mm = holaps_mm.LE_Mean
        holaps_mm.name = "Evapotranspiration"
        holaps_mm.attrs["units"] = "mm day-1 [w m-2 / 28]"
        self.update_clean_data(
            holaps_mm, msg="Transform Latent Heat (w m-2) to Evaporation (mm day-1)"
        )

        return

    def preprocess(self):
        # reproject the file from sinusoidal to WGS84 / 'ESPG:4326'
        self.reproject()
        #  chop out the correct lat/lon (changes when reprojected)
        self.chop_EA_region()
        # convert the units
        self.convert_units()
        # rename data
        self.rename_xr_object("holaps_evapotranspiration")
        # resample the time units
        self.resample_time()
        # save the netcdf file (used as reference data for MODIS and GLEAM)
        save_netcdf(
            self.clean_data, filepath=self.base_data_path / "holaps_EA_clean.nc",
            force=True
        )
        print("\n\n HOLAPS Preprocessed \n\n")
        return
