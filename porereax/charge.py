import numpy as np

from porereax.utils import BondSampler, AtomSampler


class ChargeSampler(AtomSampler):
    def __init__(self, file_name, dimension, **parameters):
        super().__init__(file_name, dimension)
        self.num_bins = parameters.get("bins", 600)
        self.range = parameters.get("range", (-3.0, 3.0))
        self.input.update({"num_bins": self.num_bins})
        self.input.update({"range": self.range})

    def init_sampling(self, atom_lib, dimension_params={}):
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "None":
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
                self.data[identifier] = {"hist": hist, "bin_edges": bin_edges}
            else:
                raise ValueError(f"Dimension {self.dimension} not supported in ChargeSampler.")
        return super().init_sampling(atom_lib, dimension_params)

    def sample(self, frame, charges, mol_index):
        print(f"Sampling charges for frame {frame}...")
        print("Molecules:", self.molecules)
        for identifier in self.molecules:
            charge = charges[mol_index[identifier]]
            hist, _ = np.histogram(charge, bins=self.num_bins, range=self.range)
            self.data[identifier]["hist"] += hist