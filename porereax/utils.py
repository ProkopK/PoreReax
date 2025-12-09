"""
Module providing utility functions for sampling molecular data.

This module defines base Sampler classes, including BondSampler and AtomSampler,
which can be extended for specific sampling tasks. It also includes functions
for saving and loading Python objects using pickle.
"""


import pickle


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