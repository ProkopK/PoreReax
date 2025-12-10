"""
### Module for sampling atomic densities.

It provides:
1. DensitySampler: A class to sample atomic densities on specified atoms and their bonds with for multiple dimensions:
    * Cartesian1D: Samples the density histogram along a specified Cartesian direction for the whole simulation box.
    * Time: Samples the number of atoms (with given bonds) per frame.
2. Functions to plot the sampled density data:
    * plot_hist: Plots density histograms from sampled data.
    * plot_time: Plots density over time from sampled data.
"""


import numpy as np
from porereax.meta_sampler import BondSampler, AtomSampler
import porereax.utils as utils
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class DensitySampler(AtomSampler):
    """
    Sampler class for atomic densities.
    """
    def __init__(self, name_out: str, dimension: str, atoms: dict, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, direction: str):
        """
        Sampler for atomic densities.

        Parameters
        ----------
        name_out : str
            Output folder name.
        dimension : str
            Sampling dimension. Supported: "Cartesian1D", "Time".
        atoms : dict
            Dictionary defining atoms to sample.
        process_id : int
            Process ID for parallel sampling.
        atom_lib : dict
            Dictionary mapping atom type strings to their type IDs.
        masses : dict
            Dictionary mapping atom type strings to their masses.
        num_frames : int
            Total number of frames to sample.
        box : np.ndarray
            Simulation box dimensions.
        num_bins : int
            Number of bins for histogram sampling (only for "Cartesian1D").
        direction : str
            Direction for histogram sampling ("x", "y", or "z") (only for "Cartesian1D").
        """
        valid_dimensions = ["Cartesian1D", "Cartesian2D", "Time"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"DensitySampler does not support dimension {dimension}")
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("DensitySampler requires a positive integer 'num_bins' parameter.")
        self.num_bins = num_bins
        self.direction = direction
        super().__init__(name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, num_bins=num_bins, direction=direction)

        # Setup data
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "Time":
                self.data[identifier] = {"densities": np.zeros(num_frames), "num_frames": 0, }
            elif self.dimension == "Cartesian1D":
                box_lengths = box
                if self.direction not in ["x", "y", "z"]:
                    raise ValueError("DensitySampler with 'Cartesian1D' dimension requires 'direction' parameter to be one of 'x', 'y', or 'z'.")
                dir_index = {"x": 0, "y": 1, "z": 2}[self.direction]
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=(0.0, box_lengths[dir_index]))
                self.data[identifier] = {"hist": hist, "bin_edges": bin_edges, "box_lengths": box_lengths, "direction": dir_index, "num_bins": num_bins, "num_frames": 0}
            elif self.dimension == "Cartesian2D":
                box_lengths = box
                if self.direction not in ["xy", "xz", "yz"]:
                    raise ValueError("DensitySampler with 'Cartesian2D' dimension requires 'direction' parameter to be one of 'xy', 'xz', or 'yz'.")
                dir_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[self.direction]
                hist, x_edges, y_edges = np.histogram2d([], [], bins=self.num_bins, range=[[0.0, box_lengths[dir_indices[0]]], [0.0, box_lengths[dir_indices[1]]]])
                self.data[identifier] = {"hist": hist, "x_edges": x_edges, "y_edges": y_edges, "box_lengths": box_lengths, "direction": dir_indices, "num_bins": num_bins, "num_frames": 0}

    def sample(self, frame: int, positions: np.ndarray, mol_index: dict):
        """
        Sample atomic densities for the given frame.

        Parameters
        ----------
        frame : int
            Current frame number.
        positions : np.ndarray
            Array of atomic positions.
        mol_index : dict
            Mapping of molecule identifiers to atom indices.
        """
        for identifier in self.molecules:
            atom_indices = mol_index[identifier]
            atom_positions = positions[atom_indices]
            self.data[identifier]["num_frames"] += 1
            if self.dimension == "Time":
                self.data[identifier]["densities"][frame] = atom_indices.shape[0]
            elif self.dimension == "Cartesian1D":
                direction = self.data[identifier]["direction"]
                hist, _ = np.histogram(atom_positions[:, direction], bins=self.data[identifier]["num_bins"], range=(0.0, self.data[identifier]["box_lengths"][direction]))
                self.data[identifier]["hist"] += hist
            elif self.dimension == "Cartesian2D":
                dir_x, dir_y = self.data[identifier]["direction"]
                hist, _, _ = np.histogram2d(atom_positions[:, dir_x], atom_positions[:, dir_y], bins=self.data[identifier]["num_bins"], range=[[0.0, self.data[identifier]["box_lengths"][dir_x]], [0.0, self.data[identifier]["box_lengths"][dir_y]]])
                self.data[identifier]["hist"] += hist

    def join_samplers(self, num_cores):
        """
        Join data from multiple samplers after parallel processing.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used.
        """
        if self.process_id != -1:
            return
        data_list = {}
        for process_id in range(num_cores):
            file_path = self.folder + f"/proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            for identifier, data in proc_data.items():
                if identifier == "input_params":
                    data_list[identifier] = data
                    continue
                elif identifier not in data_list:
                    if self.dimension == "Time":
                        data_list[identifier] = {"num_frames": np.zeros(num_cores, dtype=int),
                                                 "densities": [data["num_frames"]] * num_cores}
                    elif self.dimension == "Cartesian1D":
                        data_list[identifier] = {"num_frames": np.zeros(num_cores, dtype=int),
                                                 "hist": np.zeros((num_cores, data["num_bins"]), dtype=float),
                                                 "bin_edges": data["bin_edges"],
                                                 "box_lengths": data["box_lengths"],
                                                 "direction": data["direction"],
                                                 "num_bins": data["num_bins"]}
                    elif self.dimension == "Cartesian2D":
                        data_list[identifier] = {"num_frames": np.zeros(num_cores, dtype=int),
                                                 "hist": np.zeros((num_cores, data["num_bins"], data["num_bins"]), dtype=float),
                                                 "x_edges": data["x_edges"],
                                                 "y_edges": data["y_edges"],
                                                 "box_lengths": data["box_lengths"],
                                                 "direction": data["direction"],
                                                 "num_bins": data["num_bins"]}
                if self.dimension == "Time":
                    data_list[identifier]["num_frames"][process_id] = data["num_frames"]
                    data_list[identifier]["densities"][process_id] = data["densities"]
                elif self.dimension == "Cartesian1D":
                    data_list[identifier]["num_frames"][process_id] = data["num_frames"]
                    data_list[identifier]["hist"][process_id, :] = data["hist"]
                elif self.dimension == "Cartesian2D":
                    data_list[identifier]["num_frames"][process_id] = data["num_frames"]
                    data_list[identifier]["hist"][process_id, :, :] = data["hist"]
        combined_data = {}
        for identifier in data_list:
            if identifier == "input_params":
                combined_data[identifier] = data_list[identifier]
                continue
            combined_data[identifier] = {}
            if self.dimension == "Time":
                combined_data[identifier]["num_frames"] = np.sum(data_list[identifier]["num_frames"])
                combined_data[identifier]["densities"] = np.concatenate(data_list[identifier]["densities"])
            elif self.dimension == "Cartesian1D":
                combined_data[identifier]["num_frames"] = np.sum(data_list[identifier]["num_frames"])
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / combined_data[identifier]["num_frames"]
                combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0)
                combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"]
                combined_data[identifier]["box_lengths"] = data_list[identifier]["box_lengths"]
                combined_data[identifier]["direction"] = data_list[identifier]["direction"]
                combined_data[identifier]["num_bins"] = data_list[identifier]["num_bins"]
            elif self.dimension == "Cartesian2D":
                combined_data[identifier]["num_frames"] = np.sum(data_list[identifier]["num_frames"])
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / combined_data[identifier]["num_frames"]
                combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0)
                combined_data[identifier]["x_edges"] = data_list[identifier]["x_edges"]
                combined_data[identifier]["y_edges"] = data_list[identifier]["y_edges"]
                combined_data[identifier]["box_lengths"] = data_list[identifier]["box_lengths"]
                combined_data[identifier]["direction"] = data_list[identifier]["direction"]
                combined_data[identifier]["num_bins"] = data_list[identifier]["num_bins"]
        utils.save_object(combined_data, self.folder + "/combined.obj")


def plot_hist(link_data: str, axis=True, std=True, identifiers = [], colors = []):
    """
    Plot density histograms from sampled data.

    Parameters
    ----------
    link_data : str
        Path to the data file containing sampled density data.
    axis : bool, optional
        Whether to display axes on the plot. Default is True.
    std : bool, optional
        Whether to plot standard deviation as shaded area. Default is True.
    identifiers : list, optional
        List of molecule identifiers to plot. Default is empty list (plot all).
    colors : list, optional
        List of colors for each identifier. Default is empty list (use default colors).
    """
    fig, ax, data, identifiers, colors = utils.plot_setup(link_data, axis, identifiers, colors)
    if data["input_params"]["dimension"] != "Cartesian1D":
        return
    for i, identifier in enumerate(identifiers):
        if identifier == "input_params":
            continue
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        density_data = data[identifier]
        bin_centers = 0.5 * (density_data["bin_edges"][:-1] + density_data["bin_edges"][1:]) / 10 # Convert to nm
        hist_data = density_data["hist"]
        color = colors[i % len(colors)] if colors else None
        ax.plot(bin_centers, hist_data, label=identifier, color=color)
        if std:
            hist_std = density_data["hist_std"]
            ax.fill_between(bin_centers, 
                            hist_data - hist_std, 
                            hist_data + hist_std, 
                            color=color, 
                            alpha=0.3)
    ax.set_xlabel("Position / nm")
    ax.set_ylabel("Density / atoms")

def plot_time(link_data: str, axis: Axes | bool=True, identifiers = [], colors = [], dt=20):
    """
    Plot density over time from sampled data.

    Parameters
    ----------
    link_data : str
        Path to the data file containing sampled density data.
    axis : matplotlib.axes.Axes or bool, optional
        Axis to plot on or True to create a new one. Default is True.
    identifiers : list, optional
        List of molecule identifiers to plot. Default is empty list (plot all).
    colors : list, optional
        List of colors for each identifier. Default is empty list (use default colors).
    dt : float, optional
        Time step between frames. Default is 0.5fs
    """
    fig, ax, data, identifiers, colors = utils.plot_setup(link_data, axis, identifiers, colors)
    if data["input_params"]["dimension"] != "Time":
        return
    for i, identifier in enumerate(identifiers):
        if identifier == "input_params":
            continue
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        time_data = data[identifier]
        time_points = np.arange(0, time_data["num_frames"] * dt, dt) / 1000  # Convert to ps
        density_data = time_data["densities"]
        color = colors[i % len(colors)] if colors else None
        ax.plot(time_points, density_data, label=identifier, color=color)
    ax.set_xlabel("Time / ps")
    ax.set_ylabel("Density / atoms")

def plot_2d_hist(link_data: str, identifier: str, transpose: bool=False):
    """
    Plot 2D density histogram from sampled data.

    Parameters
    ----------
    link_data : str
        Path to the data file containing sampled density data.
    identifier : str
        Molecule identifier to plot.
    """
    data = utils.load_object(link_data)
    if data["input_params"]["dimension"] != "Cartesian2D":
        return
    if identifier not in data:
        print(f"Warning: Identifier {identifier} not found in data.")
        return
    density_data = data[identifier]
    x_edges = density_data["x_edges"] / 10  # Convert to nm
    y_edges = density_data["y_edges"] / 10  # Convert to nm
    hist = density_data["hist"]
    if transpose:
        hist = hist

    X, Y = np.meshgrid(x_edges, y_edges)
    if transpose:
        X, Y = Y, X
    plt.figure()
    plt.pcolormesh(X, Y, hist.T, shading='auto')
    if transpose:
        plt.xlabel(f"{['x','y','z'][density_data['direction'][1]]} Position / nm")
        plt.ylabel(f"{['x','y','z'][density_data['direction'][0]]} Position / nm")
    else:
        plt.xlabel(f"{['x','y','z'][density_data['direction'][0]]} Position / nm")
        plt.ylabel(f"{['x','y','z'][density_data['direction'][1]]} Position / nm")
    plt.axis('scaled')
    plt.colorbar(label='Density / atoms')
    plt.show()