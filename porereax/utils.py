"""
Module providing utility functions for sampling molecular data.

This module defines base Sampler classes, including BondSampler and AtomSampler,
which can be extended for specific sampling tasks. It also includes functions
for saving and loading Python objects using pickle.
"""


import pickle
from matplotlib.axes import Axes


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
    
def plot_setup(link_data: str, axis: Axes | bool=True, identifiers=[], colors=[]):
    """
    Set up a matplotlib figure and axis for plotting.

    Parameters
    ----------
    link_data : str
        Path to the data file to be loaded.
    axis : matplotlib.axes.Axes or bool, optional
        Axis to plot on or True to create a new one. Default is True.
    identifiers : list, optional
        List of identifiers to plot (default is an empty list, which means all identifiers).
    colors : list, optional
        List of colors to use for plotting (default is an empty list, which uses default colors
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The created axes object.
    data : dict
        The loaded data from the specified file.
    identifiers : list
        The list of identifiers to be plotted.
    colors : list
        The list of colors to be used for plotting.
    """
    import matplotlib.pyplot as plt
    data = load_object(link_data)
    if data["input_params"]["dimension"] != "Time":
        return
    if axis == True:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = axis
    identifiers = data.keys() if identifiers == [] else identifiers
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == [] else colors

    return fig, ax, data, identifiers, colors