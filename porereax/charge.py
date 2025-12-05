import numpy as np
import itertools

from porereax.utils import BondSampler, AtomSampler


class ChargeSampler(AtomSampler):
    # TODO dimension parameters handling: here does not work, must be handled in init_sampling
    def __init__(self, file_name, dimension, **parameters):
        super().__init__(file_name, dimension)
        self.num_bins = parameters.get("bins", 600)
        self.range = parameters.get("range", (-3.0, 3.0))
        self.input.update({"file_name": file_name})
        self.input.update({"dimension": dimension})
        self.input.update({"num_bins": self.num_bins})
        self.input.update({"range": self.range})

    def add_atom(self, atom, bonds=None):
        super().add_atom(atom, bonds)

    def init_sampling(self, atom_lib):
        for identifier, bonds_info in self.molecules.items():
            atom = bonds_info["atom"]
            if atom in atom_lib:
                atom = atom_lib[atom]
            else:
                raise ValueError(f"Error in ChargeSampler: Atom {atom} not found in atom library.")
            bonds = bonds_info["bonds"]
            bond_types = []
            for bond in bonds:
                if bond in atom_lib:
                    bond_types.append(atom_lib[bond])
                else:
                    raise ValueError(f"Error in ChargeSampler: Bonded atom {bond} not found in atom library.")
            bonds = list(map(list, itertools.permutations(bond_types)))
            if self.dimension == "None":
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
                self.data[identifier] = {"hist": hist, "bin_edges": bin_edges}
                self.molecules[identifier].update({"atom": atom, "bonds": bonds})
            else:
                raise ValueError(f"Dimension {self.dimension} not supported in ChargeSampler.")
    
    def sample(self, frame, charges, types, bond_topology, bond_enum):
        for identifier, bonds_info in self.molecules.items():
            atom_type = bonds_info["atom"]
            permutated_bonds = bonds_info["bonds"]
            idx = np.where(types == atom_type)[0]
            if permutated_bonds[0] != []:
                for i in idx:
                    bonds = list(bond_enum.bonds_of_particle(i))
                    particles = bond_topology[bonds].flatten()
                    other_particles = particles[particles != i]
                    other_types = list(types[other_particles])
                    if other_types in permutated_bonds:
                        charge = charges[i]
                        hist, _ = np.histogram([charge], bins=self.num_bins, range=self.range)
                        self.data[identifier]["hist"] += hist
            else:
                charge = charges[idx]
                hist, _ = np.histogram(charge, bins=self.num_bins, range=self.range)
                self.data[identifier]["hist"] += hist