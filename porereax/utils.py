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
    if axis == True:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = axis
    identifiers = data.keys() if identifiers == [] else identifiers
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == [] else colors

    return fig, ax, data, identifiers, colors

def plot_hist(axis: Axes, identifier: str, bin_edges: np.ndarray, hist_data: np.ndarray, color: str, plot_kwargs: dict, std_data: np.ndarray = None, mean_data: float = None):
    plot_kwargs["color"] = color
    plot_kwargs['label'] = identifier
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axis.plot(bin_centers, hist_data, **plot_kwargs)
    if std_data is not None:
        upper_bound = hist_data + std_data
        lower_bound = hist_data - std_data
        axis.fill_between(bin_centers, lower_bound, upper_bound, color=color, alpha=0.3)
    if mean_data is not None:
        axis.axvline(mean_data, linestyle="--", color=color, label=f"Mean {identifier}")

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