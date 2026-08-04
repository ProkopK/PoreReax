"""
Module for plotting sampled data.

The module provides functions to plot histograms, time series, and 2D density data from sampled data.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

import porereax.utils as utils


def _plot_one_line(axis: Axes, identifier: str, bin_edges: np.ndarray, hist_data: np.ndarray, color: str, plot_kwargs: dict, std_data: np.ndarray = None, mean_data: float = None, mean_std: float = None):
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
    mean_std : float, optional
        Standard deviation of the mean (default is None).
    """
    plot_kwargs["color"] = color
    plot_kwargs["label"] = identifier
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axis.plot(bin_centers, hist_data, **plot_kwargs)
    if std_data is not None:
        upper_bound = hist_data + std_data
        lower_bound = hist_data - std_data
        axis.fill_between(bin_centers,
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.3)
    if mean_data is not None:
        axis.axvline(mean_data, linestyle="--", color=color, label=f"Mean {identifier}")
    if mean_std is not None:
        axis.fill_betweenx(
            axis.get_ylim(),
            mean_data - mean_std,
            mean_data + mean_std,
            color=color,
            alpha=0.2
        )

def _plot_parameters(input_params: dict, mean: bool, density: bool):
    """
    Determine the appropriate x and y labels, density normalization, and flags for mean and density based on the sampler type and dimension.

    Parameters
    ----------
    input_params : dict
        Input parameters from the data file.
    mean : bool
        Whether to plot mean values as vertical lines.
    density : bool
        Whether to normalize histograms to density.

    Returns
    -------
    tuple
        A tuple containing x_label, y_label, density_normalization, mean, and density.
    """
    sampler_type = input_params["sampler_type"]
    if sampler_type == "ChargeSampler":
        x_label = "Charge / e"
        if density:
            y_label = "Counts per Atom"
        else:
            y_label = "Counts"
        density_normalization = "num_atoms"
    elif sampler_type == "AngleSampler":
        x_label = "Angle / degrees"
        if density:
            y_label = "Counts per Angle"
        else:
            y_label = "Counts"
        density_normalization = "num_angles"
    elif sampler_type == "BondLengthSampler":
        if input_params["dimension"] == "Bond Order":
            x_label = "Bond Order"
        elif input_params["dimension"] == "Bond Length":
            x_label = "Bond Length / Angstrom"
        if density:
            y_label = "Counts per Bond"
        else:
            y_label = "Counts"
        density_normalization = "num_bonds"
    elif sampler_type == "BondDensitySampler" or sampler_type == "DensitySampler" or sampler_type == "ReactionSampler":
        x_label = f"{input_params['direction']} Position / nm"
        y_label = "Density / atoms"
        density_normalization = None
        mean = False
        density = False
    elif sampler_type == "RdfSampler":
        x_label = "Distance r / Å"
        y_label = "g(r)"
        density_normalization = None
        mean = False
        density = False
    else:
        raise ValueError(f"Plotting is not implemented for sampler type {sampler_type} with dimension {input_params['dimension']}.")

    return x_label, y_label, density_normalization, mean, density


def _plot_hist(axis: Axes, data: dict, input_params: dict, identifiers: list, colors: list, std: bool, mean: bool, density: bool, plot_kwargs: dict):
    """
    Plot histograms for the given identifiers on the provided axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot on.
    data : dict
        Data dictionary containing histogram data for each identifier.
    input_params : dict
        Input parameters from the data file.
    identifiers : list
        List of identifiers to plot.
    colors : list
        List of colors to use for plotting.
    std : bool
        Whether to plot standard deviation shading.
    mean : bool
        Whether to plot mean values as vertical lines.
    density : bool
        Whether to normalize histograms to density.
    plot_kwargs : dict
        Additional keyword arguments for the plot function.
    """
    x_label, y_label, density_normalization, mean, density = _plot_parameters(input_params, mean, density)

    for i, identifier in enumerate(identifiers):
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        bin_edges = data[identifier]["bin_edges"]
        hist = data[identifier]["hist"]
        if density:
            hist = hist / data[identifier][density_normalization]
        hist_std = data[identifier]["hist_std"] if std else None
        mean_value = data[identifier]["mean"] if mean else None
        mean_std = data[identifier]["mean_std"] if std and mean else None
        _plot_one_line(axis, identifier, bin_edges, hist, colors[i % len(colors)], plot_kwargs, hist_std, mean_value, mean_std)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

def _plot_2d(axis: Axes, data: dict, identifier: str, transpose: bool, plot_kwargs: dict):
    """
    Plot 2D density data for the given identifier on the provided axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot on.
    data : dict
        Data dictionary containing 2D density data for each identifier.
    identifier : str
        Identifier for the data to plot.
    transpose : bool
        Whether to transpose the axes for the 2D density plot.
    plot_kwargs : dict
        Additional keyword arguments for the pcolormesh function.
    """
    if identifier not in data:
        raise ValueError(f"Identifier {identifier} not found in data.")
    density_data = data[identifier]
    x_edges = density_data["x_edges"] / 10  # Convert to nm
    y_edges = density_data["y_edges"] / 10  # Convert to nm
    hist = density_data["hist"]

    shading = plot_kwargs.pop("shading", "auto")

    X, Y = np.meshgrid(x_edges, y_edges)
    if transpose:
        X, Y = Y, X
    c = axis.pcolormesh(X, Y, hist.T, shading=shading, **plot_kwargs)
    plt.colorbar(c, ax=axis, label='Density / Counts per frame')
    if transpose:
        axis.set_xlabel(f"{['x','y','z'][density_data['direction'][1]]} / nm")
        axis.set_ylabel(f"{['x','y','z'][density_data['direction'][0]]} / nm")
    else:
        axis.set_xlabel(f"{['x','y','z'][density_data['direction'][0]]} / nm")
        axis.set_ylabel(f"{['x','y','z'][density_data['direction'][1]]} / nm")
    axis.set_aspect('equal', adjustable='box')

def _plot_time(axis: Axes, data: dict, identifiers: list, colors: list, dt: int):
    """
    Plot time series data for the given identifiers on the provided axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot on.
    data : dict
        Data dictionary containing time series data for each identifier.
    identifiers : list
        List of identifiers to plot.
    colors : list
        List of colors to use for plotting.
    dt : int
        Time step in femtoseconds for time series plots.
    """
    for i, identifier in enumerate(identifiers):
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        time_data = data[identifier]
        time_points = np.arange(0, time_data["num_frames"] * dt, dt) / 1000  # Convert to ps
        density_data = time_data["densities"]
        color = colors[i % len(colors)] if colors else None
        axis.plot(time_points, density_data, label=identifier, color=color)
    axis.set_xlabel("Time / ps")
    axis.set_ylabel("Counts per Frame")

def _plot_mol_structure(axis: Axes, data: dict, identifier: str):
    """
    Plot molecule structure counts for the given identifier on the provided axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to plot on.
    data : dict
        Data dictionary containing molecule structure counts for each identifier.
    identifier : str
        Identifier for the data to plot.
    """
    if identifier not in data:
        raise ValueError(f"Identifier {identifier} not found in data.")
    structure_counts = data[identifier]
    structures = list(structure_counts.keys())
    counts = list(structure_counts.values())
    axis.bar(structures, counts)
    axis.set_xlabel("Molecule Structure")
    axis.xaxis.set_tick_params(rotation=90)
    axis.set_ylabel("Average Count per Frame")

def plot(link_data: str, axis: Axes | None = None, identifiers: list = [], colors: list = [], std: bool = False, mean: bool = False, density: bool = False, dt: int = 50, transpose: bool = False, plot_kwargs_1d: dict = {}, plot_kwargs_2d: dict = {}) -> tuple[Figure | None, Axes]:
    """
    Plot sampled data from a data file.
    All types of samplers are supported. Depending on the sampler type and dimension, different types of plots will be generated.

    Parameters
    ----------
    link_data : str
        Path to the data file created by a sampler instance.
    axis : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis will be created (default is None).
    identifiers : list, optional
        List of identifiers to plot. If empty, all identifiers will be plotted (default is []).
    colors : list, optional
        List of colors to use for plotting. If empty, default colors will be used (default is []).
    std : bool, optional
        Whether to plot standard deviation shading (default is False).
    mean : bool, optional
        Whether to plot mean values as vertical lines (default is False).
    density : bool, optional
        Whether to normalize histograms to density (default is False).
    dt : int, optional
        Time step in femtoseconds for time series plots (default is 50).
    transpose : bool, optional
        Whether to transpose the axes for 2D density plots (default is False).
    plot_kwargs_1d : dict, optional
        Additional keyword arguments for 1D plots (default is {}).
    plot_kwargs_2d : dict, optional
        Additional keyword arguments for 2D plots (default is {}).

    Returns
    -------
    tuple
        A tuple containing the figure and axis objects. If an axis was provided, the figure will be None.
    """
    data = utils.load_object(link_data)
    input_params = data.pop("input_params", None)
    sampler_type = input_params["sampler_type"]

    if axis is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = axis
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if not colors else colors
    identifiers = identifiers if identifiers else list(data.keys())

    if sampler_type == "MoleculeStructureSampler":
        _plot_mol_structure(ax, data, identifiers[0] if identifiers else list(data.keys())[0])
    elif input_params["dimension"] == "Time":
        _plot_time(ax, data, identifiers, colors, dt)
    elif input_params["dimension"] == "Cartesian2D":
        _plot_2d(ax, data, identifiers[0] if identifiers else list(data.keys())[0], transpose, plot_kwargs_2d)
    else:
        _plot_hist(ax, data, input_params, identifiers, colors, std, mean, density, plot_kwargs_1d)

    return fig, ax