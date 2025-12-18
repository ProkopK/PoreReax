from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pickle

import porereax.utils as utils


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
    data = utils.load_object(link_data)
    if axis == True:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = axis
    identifiers = data.keys() if identifiers == [] else identifiers
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == [] else colors

    return fig, ax, data, identifiers, colors

def plot_one_line(axis: Axes, identifier: str, bin_edges: np.ndarray, hist_data: np.ndarray, color: str, plot_kwargs: dict, std_data: np.ndarray = None, mean_data: float = None, std_mean: float = None):
    """
    Plot a histogram curve on the given axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot on.
    identifier : str
        Identifier for the data being plotted.
    bin_edges : np.ndarray
        Edges of the histogram bins.
    hist_data : np.ndarray
        Histogram data to plot.
    color : str
        Color for the plot.
    plot_kwargs : dict
        Additional keyword arguments for the plot function.
    std_data : np.ndarray, optional
        Standard deviation data for shading (default is None).
    mean_data : float, optional
        Mean value to plot as a vertical line (default is None).
    std_mean : float, optional
        Standard deviation of the mean (default is None).
    """
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

def plot_hist(link_data: str, axis: Axes | bool=True, identifiers = [], colors = [], std=False, mean=False, density=False, plot_kwargs = {}):
    """
    Plot histogram curves from sampled data.

    Parameters
    ----------
    link_data : str
        Path to the data file created by a sampler instance.
    axis : matplotlib.axes.Axes or bool, optional
        Axis to plot on or True to create a new one. Default is True.
    identifiers : list, optional
        List of identifiers to plot (default is an empty list, which means all identifiers).
    colors : list, optional
        List of colors to use for plotting (default is an empty list, which uses default colors).
    std : bool, optional
        Whether to plot standard deviation shading (default is False).
    mean : bool, optional
        Whether to plot mean lines (default is False).
    density : bool, optional
        Whether to normalize histogram by number of samples (default is False).
    plot_kwargs : dict, optional
        Additional keyword arguments for the plot function.

    Returns
    -------
    None
    """
    fig, ax, data, identifiers, colors = plot_setup(link_data, axis, identifiers, colors)

    sampler_type = data["input_params"]["sampler_type"]
    if sampler_type not in ["BondLengthSampler", "AngleSampler", "ChargeSampler"]:
        print(f"Warning: plot_hist is not implemented for sampler type {sampler_type}.")
        return
    
    if sampler_type == "ChargeSampler":
        density_normalization = "num_atoms"
        x_label = "Charge / e"
        y_label = "Counts per Atom"
    elif sampler_type == "BondLengthSampler":
        density_normalization = "num_bonds"
        x_label = "Bond Length / Angstrom"
        y_label = "Counts per Bond"
    elif sampler_type == "AngleSampler":
        density_normalization = "num_angles"
        x_label = "Angle / degrees"
        y_label = "Counts per Angle"

    if data["input_params"]["dimension"] != "Histogram":
        return
    for i, identifier in enumerate(identifiers):
        if identifier == "input_params":
            continue
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        bin_edges = data[identifier]["bin_edges"]
        hist = data[identifier]["hist"]
        if density:
            hist = hist / data[identifier][density_normalization]
        std_hist = data[identifier]["std_hist"] if std else None
        mean_value = data[identifier]["mean"] if mean else None
        std_mean = data[identifier]["std_mean"] if std and mean else None
        plot_one_line(ax, identifier, bin_edges, hist, colors[i % len(colors)], plot_kwargs, std_hist, mean_value, std_mean)
    ax.set_xlabel(x_label)
    if density == False:
        ax.set_ylabel("Counts")
    else:
        ax.set_ylabel(y_label)
