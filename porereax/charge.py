"""
### Module for sampling atomic charges.

It provides:
1. ChargeSampler: A class to sample atomic charges on specified atoms and their bonded atoms.
2. Fuctions to plot the sampled charge data:
    * plot_hist: Plot histogram curves from sampled charge data.
"""


import numpy as np
from porereax.meta_sampler import AtomSampler
import porereax.utils as utils
from matplotlib.axes import Axes


class ChargeSampler(AtomSampler):
    """
    Sampler class for atomic charges.
    """
    def __init__(self, name_out: str, dimension: str, atoms: list, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, range: tuple):
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

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_index: dict, frame: object, bond_enum: object):
        charges = frame.particles.get("Charge").array if "Charge" in frame.particles else np.zeros(frame.particles.count)
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
        data_list = super().join_samplers(num_cores)
        combined_data = {}
        for identifier in data_list:
            if identifier == "input_params":
                combined_data["input_params"] = data_list["input_params"]
                continue
            combined_data[identifier] = {}
            if self.dimension == "Histogram":
                num_frames = np.sum(data_list[identifier]["num_frames"])
                num_atoms = np.sum(data_list[identifier]["num_atoms"])
                combined_data[identifier]["num_frames"] = num_frames
                combined_data[identifier]["num_atoms"] = num_atoms
                combined_data[identifier]["mean"] = np.sum(data_list[identifier]["mean_charge"]) / num_atoms if num_atoms > 0 else np.nan
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self.num_bins) # TODO check normalization
                combined_data[identifier]["std_mean"] = 0 # TODO: fix std calculation
                combined_data[identifier]["std_hist"] = np.std(data_list[identifier]["hist"]) # TODO: fix std calculation
                combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
        utils.save_object(combined_data, self.name_out + ".obj")
