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

## Pipeline

[Python Fire](https://github.com/google/python-fire) is used to generate a CLI.

### Data cleaning

Normalize values from the original csv file, remove null values, add a year series.

```bash
python run.py clean
```

### Data Processing

Turn the CSV into `numpy` arrays which can be input into the model.

```bash
python run.py engineer
```

### Models

So far, just a linear regression has been implemented as a baseline

```bash
python run.py train_model
```

## Results

Randomly selected test set. TODO is to split by year, so that the test set is better stratified from the training set.

| Model             | RMSE |
|:-----------------:|:----:|
|Linear Regression  |0.051 |
