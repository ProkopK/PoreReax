"""
Module for sampling atomic charges in molecular simulations.

This module defines the ChargeSampler class, which extends the AtomSampler
class to sample atomic charges based on specified atoms and their bonded
atoms. It supports histogram sampling of charges within a defined range and
number of bins.
This module provides fuctions to visualize and analyze the sampled charge data.
"""


import numpy as np
from porereax.meta_sampler import BondSampler, AtomSampler
import porereax.utils as utils
import matplotlib.pyplot as plt


class ChargeSampler(AtomSampler):
    """
    Sampler class for atomic charges.
    """
    def __init__(self, name_out: str, dimension: str, atoms: list, process_id=0, num_bins=600, range=(-3.0, 3.0)):
        """
        Initialize ChargeSampler.
        
        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Sampling dimension. Supported: "Histogram".
        atoms : list
            List of atom identifiers to sample.
        process_id : int, optional
            Process ID for parallel sampling.
        num_bins : int, optional
            Number of bins for histogram sampling.
        range : tuple, optional
            Range (min, max) for histogram sampling.
        """
        self.num_bins = num_bins
        self.range = range
        self.validate_inputs({"name_out": name_out, "dimension": dimension, "atoms": atoms, "num_bins": num_bins, "range": range})
        super().__init__(name_out, dimension, atoms, process_id, num_bins=num_bins, range=range)

    def init_sampling(self, atom_lib: dict, dimension_params={}):
        """
        Initialize sampling structures for charge sampling.

        Parameters
        ----------
        atom_lib : dict
            Library of atom types.
        dimension_params : dict, optional
            Additional parameters for dimension-specific sampling.

        Returns
        -------
        molecules : dict
            Processed molecules with atom types and bonded atom permutations.
        """
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "Histogram":
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
                self.data[identifier] = {"num_frames": 0, "num_atoms": 0, "mean_charge": 0.0, "hist": hist, "bin_edges": bin_edges, }
            else:
                raise ValueError(f"Dimension {self.dimension} not supported in ChargeSampler.")
        return super().init_sampling(atom_lib, dimension_params)

    def sample(self, frame: int, charges: np.ndarray, mol_index: dict):
        for identifier in self.molecules:
            charge = charges[mol_index[identifier]]
            hist, _ = np.histogram(charge, bins=self.num_bins, range=self.range)
            self.data[identifier]["hist"] += hist
            self.data[identifier]["num_frames"] += 1
            self.data[identifier]["num_atoms"] += mol_index[identifier].shape[0]
            self.data[identifier]["mean_charge"] += np.sum(charge)

    def join_samplers(self, num_cores):
        """
        Join sampler data from multiple processes.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used.
        """
        if self.process_id != 0:
            return
        if self.dimension == "Histogram":
            data_list = {}
            for process_id in range(num_cores):
                file_path = self.folder + f"/proc_{process_id}.pkl"
                proc_data = utils.load_object(file_path)
                for identifier, data in proc_data.items():
                    if identifier == "input_params":
                        data_list[identifier] = data
                        continue
                    elif identifier not in data_list:
                        data_list[identifier] = {"num_frames": np.zeros(num_cores, dtype=int),
                                                    "num_atoms": np.zeros(num_cores, dtype=int),
                                                    "mean_charge": np.zeros(num_cores, dtype=float),
                                                    "std_charge": np.zeros(num_cores, dtype=float),
                                                    "hist": np.zeros((num_cores, proc_data["input_params"]["num_bins"],), dtype=float),
                                                    "std_hist": np.zeros((num_cores, proc_data["input_params"]["num_bins"],), dtype=float),
                                                    "bin_edges": data["bin_edges"]}
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

    @staticmethod
    def validate_inputs(inputs: dict, atom_lib: dict = None):
        """
        Validate inputs for ChargeSampler.
        
        Parameters
        ----------
        inputs : dict
            Input parameters to validate.
        atom_lib : dict, optional
            Library of atom types for validation.
        
        Raises
        ------
        ValueError
            If any input parameter is invalid.
        """
        AtomSampler.validate_inputs(inputs, atom_lib, sampler_type="ChargeSampler")
        if inputs["dimension"] != "Histogram":
            raise ValueError(f"ChargeSampler does not support dimensions {inputs['dimension']}")
        if "num_bins" not in inputs or not isinstance(inputs["num_bins"], (int)) or inputs["num_bins"] <= 0:
            raise ValueError("ChargeSampler requires a positive integer 'num_bins' parameter.")
        if "range" not in inputs or (not isinstance(inputs["range"], (list, tuple, None)) or
                                     len(inputs["range"]) != 2 or
                                     inputs["range"][0] >= inputs["range"][1]):
            raise ValueError("ChargeSampler requires a 'range' parameter as a list or tuple of two numbers (min, max) with min < max.")
        

def plot_hist(link_data: str, axis=True, mean=True, std=True, density=False, identifiers = [], colors = []):
    """
    Plot histogram curves with optional mean lines and standard-deviation shading
    for one or more sampled atoms/molecules

    Parameters
    ----------
    link_data : str
        Path to a data file created by a `porereax.charge.ChargeSampler` instance 
        with `dimension` as "Histogram"
    axis : bool or matplotlib.axes.Axes, optional (default True)
        If True (default) a new matplotlib Figure and Axes are created. 
        If an Axes object is passed, that axes is used for plotting and no new Figure is created. 
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
    data = utils.load_object(link_data)
    if data["input_params"]["dimension"] != "Histogram":
        return
    if axis == True:
        fig, ax = plt.subplots()
    else:
        fig = None
        ax = axis
    identifiers = data.keys() if identifiers == [] else identifiers
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == [] else colors
    for i, identifier in enumerate(identifiers):
        charge_data = data[identifier]
        bin_centers = 0.5 * (charge_data["bin_edges"][:-1] + charge_data["bin_edges"][1:])
        hist_data = charge_data["hist"]
        if density:
            hist_data /= charge_data["num_atoms"]
        ax.plot(bin_centers, hist_data, label=f"{identifier}", color=colors[i])
        if std: # TODO std not working
            std_data = charge_data["std_hist"]
            if density:
                std_data /= charge_data["num_atoms"]
            ax.fill_between(bin_centers,
                            hist_data - std_data,
                            hist_data + std_data,
                            alpha=0.3,
                            color=colors[i])
        if mean:
            ax.axvline(charge_data["mean_charge"], linestyle="--", label=f"Mean {identifier}", color=colors[i])
    ax.set_xlabel("Charge")
    # ax.set_ylabel("")