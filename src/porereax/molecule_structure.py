"""
Module for sampling molecular structure statistics.

The module provides :class:`MoleculeStructureSampler` to sample bonding environments of all atomtypes and reveal the atom structures.
"""

import numpy as np
import porereax.utils as utils
import matplotlib.pyplot as plt
import os

from porereax.meta_sampler import Sampler


class MoleculeStructureSampler(Sampler):
    """
    Sampler class for molecule structure analysis.
    """

    def __init__(self, name_out: str, dimension: str, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict):
        valid_dimensions = ["MoleculeStructure"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"MoleculeStructureSampler does not support dimension {dimension}")
        super().__init__(name_out, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)

        # Setup data
        self._data["num_frames"] = 0
        self._data["structure_counts"] = {}
        for atom_type in atom_lib.values():
            self._data["structure_counts"][atom_type] = {}

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_index: dict, frame: object, bond_enum: object):
        atom_types = frame.particles.particle_types.array
        bond_topology = frame.particles.bonds.topology.array
        positions = frame.particles.positions.array
        position_mask = self._region(positions)
        for atom_type in self._data["structure_counts"]:
            atoms = np.where(atom_types == atom_type)[0]
            for atom in atoms:
                bonds = list(bond_enum.bonds_of_particle(atom))
                particles = bond_topology[bonds].flatten()
                other_particles = particles[particles != atom]
                other_types = np.sort(atom_types[other_particles])
                key = tuple(other_types)
                if position_mask[atom]:
                    if key not in self._data["structure_counts"][atom_type]:
                        self._data["structure_counts"][atom_type][key] = 0
                    self._data["structure_counts"][atom_type][key] += 1
        self._data["num_frames"] += 1

    def join_samplers(self, num_cores):
        if self._process_id != -1:
            return
        combined_data = {}
        num_frames = 0
        type_to_name = {v: k for k, v in self._atom_lib.items()}
        for process_id in range(num_cores) if num_cores > 1 else [-1]:
            file_path = self._name_out + f"_proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            os.remove(file_path)
            for identifier, data in proc_data.items():
                if identifier == "input_params":
                    combined_data["input_params"] = data
                elif identifier == "num_frames":
                    num_frames += data
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
        for atom in combined_data:
            if atom != "input_params":
                combined_data[atom] = dict(sorted(combined_data[atom].items(), key=lambda item: item[1], reverse=True))
                for structure in combined_data[atom]:
                    combined_data[atom][structure] /= num_frames

        utils.save_object(combined_data, self._name_out + ".obj")
