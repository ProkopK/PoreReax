import numpy as np
from porereax.meta_sampler import BondSampler
import porereax.utils as utils


class BondLengthSampler(BondSampler):
    """
    Sampler class for bond lengths and bond orders between bonded atom pairs.
    """
    def __init__(self, name_out: str, dimension: str, bonds: list, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, range: tuple):
        valid_dimensions = ["Bond Length", "Bond Order"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"BondLengthSampler does not support dimension {dimension}")
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("BondLengthSampler requires a positive integer 'num_bins' parameter.")
        if (not isinstance(range, (list, tuple)) or 
                len(range) != 2 or
                range[0] >= range[1]):
            raise ValueError("BondLengthSampler requires a 'range' parameter as a list or tuple of two numbers (min, max) with min < max.")
        self.num_bins = num_bins
        self.range = range
        super().__init__(name_out, dimension, bonds, process_id, atom_lib, masses, num_frames, box, num_bins=num_bins, range=range)

        # Setup data
        for identifier in self.bonds:
            hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
            if self.dimension in ["Bond Length", "Bond Order"]:
                self.data[identifier] = {"num_frames": 0, "num_bonds": 0, "mean": 0.0, "hist": hist, "bin_edges": bin_edges, }

    def sample(self, frame: int, positions: np.ndarray, bond_index: dict, bond_topology: np.ndarray, bond_orders: np.ndarray, **parameters):
        for identifier in self.bonds:
            bonds = bond_topology[bond_index[identifier]]
            if bonds.size == 0:
                continue
            position = positions[bonds]
            if self.dimension == "Bond Length":
                bond_lengths = np.linalg.norm(utils.min_image_convention(position[:, 0, :] - position[:, 1, :], self.box), axis=1)
                hist, _ = np.histogram(bond_lengths, bins=self.num_bins, range=self.range)
                self.data[identifier]["mean"] += np.sum(bond_lengths)
            if self.dimension == "Bond Order":
                bond_order = bond_orders[bond_index[identifier]]
                hist, _ = np.histogram(bond_order, bins=self.num_bins, range=self.range)
                self.data[identifier]["mean"] += np.sum(bond_order)
            self.data[identifier]["hist"] += hist
            self.data[identifier]["num_frames"] += 1
            self.data[identifier]["num_bonds"] += bonds.shape[0]

    def join_samplers(self, num_cores):
        data_list = super().join_samplers(num_cores)
        combined_data = {}
        for identifier in data_list:
            if identifier == "input_params":
                combined_data["input_params"] = data_list["input_params"]
                continue
            combined_data[identifier] = {}

            num_frames = np.sum(data_list[identifier]["num_frames"])
            num_bonds = np.sum(data_list[identifier]["num_bonds"])
            hist = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self.num_bins)
            mean = np.sum(data_list[identifier]["mean"]) / num_bonds if num_bonds > 0 else 0.0
            std_hist = np.std(data_list[identifier]["hist"], axis=0)
            combined_data[identifier]["num_frames"] = num_frames
            combined_data[identifier]["num_bonds"] = num_bonds
            combined_data[identifier]["mean"] = mean
            combined_data[identifier]["hist"] = hist
            combined_data[identifier]["std_hist"] = std_hist
            combined_data[identifier]["std_mean"] = 0 # TODO: fix std calculation
            combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
        utils.save_object(combined_data, self.folder + "/combined.obj")
