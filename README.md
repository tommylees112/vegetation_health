# vegetation_health
predicting vegetation health from precipitation and temperature

Notes about the data:
- vars_list : all of the variables that we are regridding onto a common grid.
               They are all from different sources
    [lst_day, lst_night, lst_mean, lst_mean, evap, baresoil_evap, pet, transp,
    surface_sm, rootzone_sm, sm, precip, ndvi, evi]

- East Africa is defined here as the area of the original .nc file (spi_spei.nc)
    lat min - lat max : -4.9750023 15.174995
    lon min - lon max : 32.524994 48.274994
    BoundingBox(left, bottom, right, top)
        (32.524994, -4.9750023, 15.174995, 48.274994)

- Time Range
    2010-01-01 : 2017-01-01
