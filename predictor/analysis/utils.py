import numpy as np
from pathlib import Path
from collections import namedtuple



def load_model_data(train_or_test, target='ndvi', train_dir='', test_dir=''):
    """ Return a named tuple with the following data attrs
    x, y, latlon, years. This is the data fed through the model.

    Example:
    -------
    >>> train_data = load_model_data("train")
    >>> test_data = load_model_data("test")
    """
    if train_or_test == "test":
        if test_dir == "":
            data_dir = Path('.') / "data" / "processed" /  target  / "arrays" / "test"
        else:
            data_dir = Path(test_dir)
    elif train_or_test == "train":
        if train_dir == "":
            data_dir = Path('.') / "data" / "processed" /  target  / "arrays" / "train"
        else:
            data_dir = Path(train_dir)
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

# ------------------------------------------------------------------------------
# General utils
# ------------------------------------------------------------------------------

def pickle_files(filepaths, vars):
    """ """
    assert len(filepaths) == len(vars), f"filepaths should be same size as vars because each variable needs a filepath! currently: len(filepaths): {len(filepaths)} len(vars): {len(vars)}"

    for i, filepath in enumerate(filepaths):
        save_pickle(filepath, variable)


def load_pickle(filepath):
    """ """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(filepath, variable):
    """ """
    with open(filepath, 'wb') as f:
        pickle.dump(variable, f)
    return
