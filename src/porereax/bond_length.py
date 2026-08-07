"""
Module for sampling bond lengths and bond orders.

The module provides :class:`BondLengthSampler` to sample bond length histograms (in Angstroms) or ReaxFF bond order histograms for specified bonds.
"""

import numpy as np
from porereax.meta_sampler import BondSampler
import porereax.utils as utils


class BondLengthSampler(BondSampler):
    """
    Sampler class for bond lengths and bond orders between bonded atom pairs.
    """
    def __init__(self, name_out: str, bonds: list, dimension: str, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, num_bins: int, range: tuple):
        valid_dimensions = ["Bond Length", "Bond Order"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"BondLengthSampler does not support dimension {dimension}")
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("BondLengthSampler requires a positive integer 'num_bins' parameter.")
        if (not isinstance(range, (list, tuple)) or
                len(range) != 2 or
                range[0] >= range[1]):
            raise ValueError("BondLengthSampler requires a 'range' parameter as a list or tuple of two numbers (min, max) with min < max.")
        self._num_bins = num_bins
        self._range = range
        super().__init__(name_out, bonds, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties, num_bins=num_bins, range=range)

        # Setup data
        for identifier in self._bonds:
            hist, bin_edges = np.histogram([], bins=self._num_bins, range=self._range)
            self._data[identifier] = {"num_frames": 0, "num_bonds": 0, "mean": 0.0, "hist": hist, "bin_edges": bin_edges, }

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        bond_topology = frame.particles.bonds.topology.array
        positions = frame.particles.positions.array
        position_mask = self._region(positions)
        for identifier in self._bonds:
            b_mask = bond_mask[identifier] & position_mask[bond_topology[:, 0]] & position_mask[bond_topology[:, 1]]
            bonds = bond_topology[b_mask]
            position = positions[bonds]

            if self._dimension == "Bond Length":
                bond_lengths = np.linalg.norm(utils.min_image_convention(position[:, 0, :] - position[:, 1, :], self._box), axis=1)
                hist, _ = np.histogram(bond_lengths, bins=self._num_bins, range=self._range)
                self._data[identifier]["mean"] += np.sum(bond_lengths)
            elif self._dimension == "Bond Order":
                bond_orders = frame.particles.bonds.get("Bond Order").array if "Bond Order" in frame.particles.bonds else np.zeros(frame.particles.bonds.count)
                bond_order = bond_orders[b_mask]
                hist, _ = np.histogram(bond_order, bins=self._num_bins, range=self._range)
                self._data[identifier]["mean"] += np.sum(bond_order)
            self._data[identifier]["hist"] += hist
            self._data[identifier]["num_frames"] += 1
            self._data[identifier]["num_bonds"] += bonds.shape[0]

    def join_samplers(self, num_cores):
        data_list = super().join_samplers(num_cores)
        combined_data = {}
        input_params = data_list.pop("input_params", None)
        combined_data["input_params"] = input_params
        for identifier in data_list:
            combined_data[identifier] = {}

            num_frames = np.sum(data_list[identifier]["num_frames"])
            num_bonds = np.sum(data_list[identifier]["num_bonds"])
            hist = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self._num_bins)
            mean = np.sum(data_list[identifier]["mean"]) / num_bonds if num_bonds > 0 else 0.0
            hist_std = np.std(data_list[identifier]["hist"], axis=0)
            combined_data[identifier]["num_frames"] = num_frames
            combined_data[identifier]["num_bonds"] = num_bonds
            combined_data[identifier]["mean"] = mean
            combined_data[identifier]["hist"] = hist
            combined_data[identifier]["hist_std"] = hist_std
            combined_data[identifier]["mean_std"] = 0 # TODO: fix std calculation
            combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
        utils.save_object(combined_data, self._name_out + ".obj")
