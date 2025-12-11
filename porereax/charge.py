"""
### Module for sampling atomic charges.

It provides:
1. ChargeSampler: A class to sample atomic charges on specified atoms and their bonded atoms.
2. Fuctions to plot the sampled charge data:
    * plot_hist: Plot histogram curves from sampled charge data.
"""


import numpy as np
from porereax.meta_sampler import BondSampler, AtomSampler
import porereax.utils as utils
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class ChargeSampler(AtomSampler):
    """
    Sampler class for atomic charges.
    """
    def __init__(self, name_out: str, dimension: str, atoms: dict, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, 
    range: tuple):
        """
        Sampler for atomic charges.

        Parameters
        ----------
        name_out : str
            Output folder name.
        dimension : str
            Sampling dimension. Currently only "Histogram" is supported.
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
            Number of bins for histogram sampling.
        range : tuple
            Range (min, max) for histogram sampling.
        """
        valid_dimensions = ["Histogram"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"ChargeSampler does not support dimension {dimension}")
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("ChargeSampler requires a positive integer 'num_bins' parameter.")
        if (not isinstance(range, (list, tuple)) or 
                len(range) != 2 or
                range[0] >= range[1]):
            raise ValueError("ChargeSampler requires a 'range' parameter as a list or tuple of two numbers (min, max) with min < max.")
        self.num_bins = num_bins
        self.range = range
        super().__init__(name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, num_bins=num_bins, range=range)

        # Setup data
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "Histogram":
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
                self.data[identifier] = {"num_frames": 0, "num_atoms": 0, "mean_charge": 0.0, "hist": hist, "bin_edges": bin_edges, }

    def sample(self, frame: int, charges: np.ndarray, mol_index: dict):
        """
        Sample charges for the current frame.

        Parameters
        ----------
        frame : int
            Current frame index.
        charges : np.ndarray
            Array of atomic charges.
        mol_index : dict
            Dictionary mapping molecule identifiers to atom indices.
        """
        for identifier in self.molecules:
            atom_indices = mol_index[identifier]
            atom_charges = charges[atom_indices]
            hist, _ = np.histogram(atom_charges, bins=self.num_bins, range=self.range)
            self.data[identifier]["hist"] += hist
            self.data[identifier]["num_frames"] += 1
            self.data[identifier]["num_atoms"] += mol_index[identifier].shape[0]
            self.data[identifier]["mean_charge"] += np.sum(atom_charges)

    def join_samplers(self, num_cores):
        """
        Join sampler data from multiple processes.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used.
        """
        if self.process_id != -1:
            return
        data_list = {}
        for process_id in range(num_cores) if num_cores > 1 else [-1]:
            file_path = self.folder + f"/proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            for identifier, data in proc_data.items():
                if identifier == "input_params":
                    data_list[identifier] = data
                    continue
                elif identifier not in data_list:
                    if self.dimension == "Histogram":
                        data_list[identifier] = {"num_frames": np.zeros(num_cores, dtype=int),
                                                 "num_atoms": np.zeros(num_cores, dtype=int),
                                                 "mean_charge": np.zeros(num_cores, dtype=float),
                                                 "hist": np.zeros((num_cores, proc_data["input_params"]["num_bins"],), dtype=float),
                                                 "bin_edges": data["bin_edges"]}
                if self.dimension == "Histogram":
                    data_list[identifier]["num_frames"][process_id] = data["num_frames"]
                    data_list[identifier]["num_atoms"][process_id] = data["num_atoms"]
                    data_list[identifier]["mean_charge"][process_id] = data["mean_charge"] / data["num_atoms"] if data["num_atoms"] > 0 else np.nan 
                    data_list[identifier]["hist"][process_id, :] = data["hist"] / data["num_frames"] if data["num_frames"] > 0 else data["hist"]
        combined_data = {}
        for identifier in data_list:
            if identifier == "input_params":
                combined_data[identifier] = data_list[identifier]
                continue
            combined_data[identifier] = {}
            if self.dimension == "Histogram":
                combined_data[identifier]["mean_charge"] = np.average(data_list[identifier]["mean_charge"], weights=data_list[identifier]["num_frames"])
                combined_data[identifier]["std_charge"] = np.sqrt(np.average((data_list[identifier]["mean_charge"] - combined_data[identifier]["mean_charge"])**2, weights=data_list[identifier]["num_frames"]))
                combined_data[identifier]["hist"] = np.average(data_list[identifier]["hist"], axis=0, weights=data_list[identifier]["num_frames"])
                combined_data[identifier]["std_hist"] = np.average((data_list[identifier]["hist"] - combined_data[identifier]["hist"])**2, axis=0, weights=data_list[identifier]["num_frames"])
                combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"]
                combined_data[identifier]["num_atoms"] = np.sum(data_list[identifier]["num_atoms"])
                combined_data[identifier]["num_frames"] = np.sum(data_list[identifier]["num_frames"])
        else:
            pass
        utils.save_object(combined_data, self.folder + "/combined.obj")

def plot_hist(link_data: str, axis=True, mean=True, std=True, density=False, identifiers = [], colors = []):
    """
    Plot histogram curves with optional mean lines and standard-deviation shading
    for one or more sampled atoms/molecules

    Parameters
    ----------
    link_data : str
        Path to a data file created by a `porereax.charge.ChargeSampler` instance 
        with `dimension` as "Histogram"
    axis : matplotlib.axes.Axes or bool, optional
        Axis to plot on or True to create a new one. Default is True.
    mean : bool, optional (default True)
        If True plot a vertical dashed line at each atom's "mean_charge".
    std : bool, optional (default True)
        If True include a shaded region as the standard deviation
    density : bool, optional (default False)
        If True scale the y-axis by the number of atoms to convert counts into a per-atom
        density.
    identifiers : list, optional (default [])
        List of molecules to plot. If left as the default empty list,
        sampled molecules will be plotted in iteration order.
    colors : sequence, optional (default [])
        Sequence of colors used to plot. If left empty the default color cycle from
        matplotlib (plt.rcParams['axes.prop_cycle']) is used.
    """
    fig, ax, data, identifiers, colors = utils.plot_setup(link_data, axis, identifiers, colors)
    if data["input_params"]["dimension"] != "Histogram":
        return
    for i, identifier in enumerate(identifiers):
        if identifier == "input_params":
            continue
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        charge_data = data[identifier]
        bin_centers = 0.5 * (charge_data["bin_edges"][:-1] + charge_data["bin_edges"][1:])
        hist_data = charge_data["hist"]
        if density:
            hist_data /= charge_data["num_atoms"]
        color = colors[i % len(colors)] if colors else None
        ax.plot(bin_centers, hist_data, label=f"{identifier}", color=color)
        if std: # TODO std not working
            std_data = charge_data["std_hist"]
            if density:
                std_data /= charge_data["num_atoms"]
            ax.fill_between(bin_centers,
                            hist_data - std_data,
                            hist_data + std_data,
                            alpha=0.3,
                            color=color)
        if mean:
            ax.axvline(charge_data["mean_charge"], linestyle="--", label=f"Mean {identifier}", color=colors[i])
    ax.set_xlabel("Charge")
    # ax.set_ylabel("") # TODO y-label name