"""
Module for sampling atomic charges in molecular simulations.

This module defines the ChargeSampler class, which extends the AtomSampler
class to sample atomic charges based on specified atoms and their bonded
atoms. It supports histogram sampling of charges within a defined range and
number of bins.
This module provides fuctions to visualize and analyze the sampled charge data.
"""


import numpy as np
from porereax.utils import BondSampler, AtomSampler


class ChargeSampler(AtomSampler):
    """
    Sampler class for atomic charges.
    """
    def __init__(self, link_out: str, dimension: str, atoms: list, process_id=0, num_bins=600, range=(-3.0, 3.0)):
        """
        Initialize ChargeSampler.
        
        Parameters
        ----------
        link_out : str
            Output file link.
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
        self.validate_inputs({"link_out": link_out, "dimension": dimension, "atoms": atoms, "num_bins": num_bins, "range": range})
        super().__init__(link_out, dimension, atoms, process_id, num_bins=num_bins, range=range)

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
                self.data[identifier] = {"hist": hist, "bin_edges": bin_edges}
            else:
                raise ValueError(f"Dimension {self.dimension} not supported in ChargeSampler.")
        return super().init_sampling(atom_lib, dimension_params)

    def sample(self, frame: int, charges: np.ndarray, mol_index: dict):
        for identifier in self.molecules:
            charge = charges[mol_index[identifier]]
            hist, _ = np.histogram(charge, bins=self.num_bins, range=self.range)
            self.data[identifier]["hist"] += hist

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