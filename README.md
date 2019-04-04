# Vegetation Health
predicting vegetation health from precipitation and temperature

Notes about the data:
- The following variables are used by the model: `['lst_night', 'lst_day', 'precip', 'sm', 'ndvi', 'evi', 'ndvi_anomaly']`.
They are all from different sources


- East Africa is defined here as the area of the original `.nc` file (`spi_spei.nc`)
    
    lat min,  lat max : `-4.9750023`,  `15.174995`
    
    lon min, lon max : `32.524994`,  `48.274994`

    This makes the following bounding box: (left, bottom, right, top):  `(32.524994, -4.9750023, 15.174995, 48.274994)`

## Pipeline

[Python Fire](https://github.com/google/python-fire) is used to generate a CLI.

### Data cleaning

Normalize values from the original csv file, remove null values, add a year series.

```bash
python run.py clean
```
A target can be selected from the variables defined above by adding the flag `--target`, e.g. 
`--target=ndvi_anomaly`. By default, the target is `ndvi`.

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

Models are trained on data before 2016, and evaluated on 2016 data. Vegetation health in June is being predicted.

In addition, vegetation health can be hidden from the model to better understand the effects of the other features.

| Model                    | RMSE | RMSE (no veg) |
|:------------------------:|:----:|:-------------:|
|Linear Regression         |0.040 |0.084          |
|Feedforward neural network|0.038 |0.070          |
|Recurrent neural network  |0.035 |0.060          |
