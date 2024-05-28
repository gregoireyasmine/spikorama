import json
import pandas as pd
import numpy as np


def jload_ignore_na(item):
    """
    Return numpy array corresponding to a JSON string.
    If the string is not a valid JSON string, considers it as a normal string and returns the string
    If the item is not a string (e.g. a N/A value), returns the item.
    """
    if type(item) == str:
        try:
            return np.array(json.loads(item))
        except ValueError:
            return item
    else:
        return item


def jdump_ignore_na(item):
    """
    Return the JSON string corresponding to a list/ndarray.
    If the item is not a list/ndarray (typically a N/A value), returns the item.
    """
    return json.dumps(item) if type(item) == list else json.dumps(list(item.astype(np.float64))) if type(item) == np.ndarray else item


def load_data(path):
    """Loads a CSV and converts JSON elements strings to numpy arrays."""
    df = pd.read_csv(path, sep=';')
    for k in df.keys():
        df[k] = df[k].apply(jload_ignore_na)
    return df


def save_data(df, path):
    """Saves a CSV after converting lists or numpy arrays to JSON strings"""
    for k in df.keys():
        df[k] = df[k].apply(jdump_ignore_na)
    df.to_csv(path, sep=';')


