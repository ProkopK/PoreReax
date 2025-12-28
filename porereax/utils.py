"""
Module providing utility functions for sampling molecular data.

This module defines base Sampler classes, including BondSampler and AtomSampler,
which can be extended for specific sampling tasks. It also includes functions
for saving and loading Python objects using pickle.
"""


import pickle
from matplotlib.axes import Axes
import numpy as np


def save_object(obj, filename):
    """
    Save a Python object to a file using pickle.

    Parameters
    ----------
    obj : any
        The Python object to be saved.
    filename : str
        The path to the file where the object will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filename):
    """
    Load a Python object from a file using pickle.

    Parameters
    ----------
    filename : str
        The path to the file from which the object will be loaded.

    Returns
    -------
    any
        The loaded Python object.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def min_image_convention(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Apply the minimal image convention to a vector given the simulation box dimensions.

    Parameters
    ----------
    vec : np.ndarray
        The input vector (shape: (N, 3)).
    box : np.ndarray
        The simulation box dimensions (shape: (3,)).

    Returns
    -------
    np.ndarray
        The vector adjusted by the minimal image convention (shape: (N, 3)).
    """
    return vec - box * np.round(vec / box)

def get_identifiers(link_data: str):
    """
    Retrieve the list of identifiers from a data file.

    Parameters
    ----------
    link_data : str
        Path to the data file created by a sampler instance.

    Returns
    -------
    list
        List of identifiers present in the data file.
    """
    data = load_object(link_data)
    return [identifier for identifier in data.keys() if identifier != "input_params"]