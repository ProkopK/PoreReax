import numpy as np
import porereax.utils as utils
from porereax.meta_sampler import Sampler
import matplotlib.pyplot as plt


class BondStructureSampler(Sampler):
    """
    Sampler class for bond structure analysis.
    """
    def __init__(self, name_out: str, dimension: str, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray):
        valid_dimensions = ["BondStructure"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"BondStructureSampler does not support dimension {dimension}")
        super().__init__(name_out, dimension, process_id, atom_lib, masses, num_frames, box)

        # Setup data
        if self.dimension == "BondStructure":
            self.data = {"num_frames": 0, "structure_counts": {}}
            for atom_type in atom_lib.values():
                self.data["structure_counts"][atom_type] = {}

    def sample(self, frame: int, bond_enum, bond_topology: np.ndarray, atom_types: np.ndarray):
        if self.dimension == "BondStructure":
            for atom_type in self.data["structure_counts"]:
                atoms = np.where(atom_types == atom_type)[0]
                for atom in atoms:
                    bonds = list(bond_enum.bonds_of_particle(atom))
                    particles = bond_topology[bonds].flatten()
                    other_particles = particles[particles != atom]
                    other_types = np.sort(atom_types[other_particles])
                    key = tuple(other_types)
                    if key not in self.data["structure_counts"][atom_type]:
                        self.data["structure_counts"][atom_type][key] = 0
                    self.data["structure_counts"][atom_type][key] += 1
            self.data["num_frames"] += 1

    def join_samplers(self, num_cores):
        if self.process_id != -1:
            return
        combined_data = {}
        type_to_name = {v: k for k, v in self.atom_lib.items()}
        for process_id in range(num_cores) if num_cores > 1 else [-1]:
            file_path = self.folder + f"/proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            for identifier, data in proc_data.items():
                if identifier == "input_params":
                    combined_data[identifier] = data
                elif identifier == "num_frames":
                    if identifier not in combined_data:
                        combined_data[identifier] = 0
                    combined_data[identifier] += data
                elif identifier == "structure_counts":
                    for key, value in data.items():
                        atom = type_to_name[key]
                        if atom not in combined_data:
                            combined_data[atom] = {}
                        for structure, count in value.items():
                            name = atom + "+" + "_".join([type_to_name[t] for t in structure])
                            if name not in combined_data[atom]:
                                combined_data[atom][name] = 0
                            combined_data[atom][name] += count
        utils.save_object(combined_data, self.folder + "/combined.obj")


def plot(link_data: str, identifier):
    data = utils.load_object(link_data)
    num_frames = data["num_frames"]
    if identifier not in data:
        raise ValueError(f"Identifier {identifier} not found in data.")
    structure_counts = data[identifier]
    structures = list(structure_counts.keys())
    counts = np.array([structure_counts[s] for s in structures]) / num_frames
    fig, ax = plt.subplots()
    ax.bar(structures, counts)
    ax.set_xlabel("Bond Structure")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_ylabel("Average Count per Frame")