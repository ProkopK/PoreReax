"""
Module for sampling atomic charges.

The module provides :class:`ChargeSampler` to sample charge histograms of specified atom structures.
"""


import numpy as np
import porereax.utils as utils

from porereax.meta_sampler import AtomSampler, _ATOM_SAMPLER_INIT_PARAMS
from porereax.utils import Substitution


@Substitution(params=_ATOM_SAMPLER_INIT_PARAMS)
class ChargeSampler(AtomSampler):
    """
    Sampler class for atomic charges.

    Parameters
    ----------
    %(params)s
    range : tuple
        Range (min, max) for histogram sampling.
    """
    def __init__(self, name_out: str, atoms: list, dimension: str, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, range: tuple):
        valid_dimensions = ["Histogram"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"ChargeSampler does not support dimension {dimension}")
        if (not isinstance(range, (list, tuple)) or 
                len(range) != 2 or
                range[0] >= range[1]):
            raise ValueError("ChargeSampler requires a 'range' parameter as a list or tuple of two numbers (min, max) with min < max.")
        self._min_range, self._max_range = range
        self._min_range = np.round(self._min_range * 1000)
        self._max_range = np.round(self._max_range * 1000)
        self._range = (self._min_range, self._max_range)
        self._num_bins = int(self._max_range - self._min_range)
        super().__init__(name_out, atoms, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        self._input.update({
            "num_bins": self._num_bins,
            "range": range,
        })

        # Setup data
        for identifier, bonds_info in self._molecules.items():
            hist, bin_edges = np.histogram([], bins=self._num_bins, range=self._range)
            self._data[identifier] = {"num_frames": 0, "num_atoms": 0, "mean_charge": 0.0, "hist": hist, "bin_edges": bin_edges, }

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        charges = frame.particles.get("Charge").array if "Charge" in frame.particles else np.zeros(frame.particles.count)
        charges = np.round(charges * 1000)
        positions = frame.particles.positions.array
        position_mask = self._region(positions)
        for identifier in self._molecules:
            mol_mask = mol_index[identifier] & position_mask
            atom_charges = charges[mol_mask]
            hist, _ = np.histogram(atom_charges, bins=self._num_bins, range=self._range)
            self._data[identifier]["hist"] += hist
            self._data[identifier]["num_frames"] += 1
            self._data[identifier]["num_atoms"] += atom_charges.shape[0]
            self._data[identifier]["mean_charge"] += np.sum(atom_charges)

    def join_samplers(self, num_cores: int) -> None:
        data_list = super()._collect_sampler_data(num_cores)
        combined_data = {}
        input_params = data_list.pop("input_params", None)
        combined_data["input_params"] = input_params
        for identifier in data_list:
            combined_data[identifier] = {}
            if self._dimension == "Histogram":
                num_frames = np.sum(data_list[identifier]["num_frames"])
                num_atoms = np.sum(data_list[identifier]["num_atoms"])
                combined_data[identifier]["num_frames"] = num_frames
                combined_data[identifier]["num_atoms"] = num_atoms
                combined_data[identifier]["mean"] = np.sum(data_list[identifier]["mean_charge"]) / num_atoms if num_atoms > 0 else np.nan
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self._num_bins) # TODO check normalization
                combined_data[identifier]["mean_std"] = 0 # TODO: fix std calculation
                combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"]) # TODO: fix std calculation
                combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0] / 1000.0
        utils.save_object(combined_data, self._name_out + ".obj")
