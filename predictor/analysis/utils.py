import numpy as np
from pathlib import Path
from collections import namedtuple



def load_model_data(train_or_test, target='ndvi'):
    """ Return a named tuple with the following data attrs
    x, y, latlon, years. This is the data fed through the model.
    """
    if train_or_test == "test":
        data_dir = Path('.') / "data" / "processed" /  target  / "arrays" / "test"
    elif train_or_test == "train":
        data_dir = Path('.') / "data" / "processed" /  target  / "arrays" / "train"
    else:
        assert False, "train_or_test must be either ['train','test']"

    Data = namedtuple('Data',["x","y","latlon","years"])
    data = Data(
        x=np.load(data_dir/"x.npy"),
        y=np.load(data_dir/"y.npy"),
        latlon=np.load(data_dir/"latlon.npy"),
        years=np.load(data_dir/"years.npy"),
    )

    return data
