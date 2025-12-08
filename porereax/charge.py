import numpy as np

from porereax.utils import BondSampler, AtomSampler


class ChargeSampler(AtomSampler):
    def __init__(self, link_out: str, dimension: str, atoms: list, process_id=0, num_bins=600, range=(-3.0, 3.0)):
        self.num_bins = num_bins
        self.range = range
        self.validate_inputs({"link_out": link_out, "dimension": dimension, "atoms": atoms, "num_bins": num_bins, "range": range})
        super().__init__(link_out, dimension, atoms, process_id, num_bins=num_bins, range=range)

    def init_sampling(self, atom_lib: dict, dimension_params={}):
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "None":
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
        AtomSampler.validate_inputs(inputs, atom_lib, sampler_type="ChargeSampler")
        if inputs["dimension"] != "None":
            raise ValueError(f"ChargeSampler only supports 'None' dimension, got {inputs['dimension']}.")
        if "num_bins" not in inputs or not isinstance(inputs["num_bins"], (int)) or inputs["num_bins"] <= 0:
            raise ValueError("ChargeSampler requires a positive integer 'num_bins' parameter.")
        if "range" not in inputs or (not isinstance(inputs["range"], (list, tuple, None)) or
                                     len(inputs["range"]) != 2 or
                                     inputs["range"][0] >= inputs["range"][1]):
            raise ValueError("ChargeSampler requires a 'range' parameter as a list or tuple of two numbers (min, max) with min < max.")