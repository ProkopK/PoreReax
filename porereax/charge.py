import numpy as np

from porereax.utils import BondSampler, AtomSampler


class ChargeSampler(AtomSampler):
    def __init__(self, link_out, dimension, atoms, process_id=0, **parameters):
        if parameters.get("num_bins") is None:
            parameters["num_bins"] = 600
        if not isinstance(parameters["num_bins"], int) or parameters["num_bins"] <= 0:
            raise ValueError("num_bins must be a positive integer.")
        if parameters.get("range") is None:
            parameters["range"] = (-3.0, 3.0)
        if (not isinstance(parameters["range"], (list, tuple)) or
                len(parameters["range"]) != 2 or
                parameters["range"][0] >= parameters["range"][1]):
            raise ValueError("range must be a list or tuple of two numbers (min, max) with min < max.")
        super().__init__(link_out, dimension, atoms, process_id, **parameters)
        self.num_bins = parameters["num_bins"] 
        self.range = parameters["range"]

    def init_sampling(self, atom_lib, dimension_params={}):
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "None":
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
                self.data[identifier] = {"hist": hist, "bin_edges": bin_edges}
            else:
                raise ValueError(f"Dimension {self.dimension} not supported in ChargeSampler.")
        return super().init_sampling(atom_lib, dimension_params)

    def sample(self, frame, charges, mol_index):
        for identifier in self.molecules:
            charge = charges[mol_index[identifier]]
            hist, _ = np.histogram(charge, bins=self.num_bins, range=self.range)
            self.data[identifier]["hist"] += hist