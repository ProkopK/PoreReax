"""
Module for sampling molecular structure statistics.

The module provides :class:`MoleculeStructureSampler` to sample bonding
environments of all atomtypes and reveal the atom structures.
"""

import numpy as np

from porereax.meta_sampler import _NAME_OUT_PARAM, _SAMPLER_INIT_PARAMS, Sampler
from porereax.utils import Substitution, _neighbor_atoms_excluding, save_object


def _build_structure_key(node, parent, remaining, atom_types, bond_topology, bond_enum):
    """
    Recursively build a hashable, sortable key describing the bonding
    environment out to `remaining` steps from `node` (excluding the edge back
    to `parent`).

    At the deepest expanded shell (`remaining <= 1`) the key is a flat tuple
    of sorted neighbour type IDs, matching the original 1-hop behaviour.
    Otherwise it is a tuple of sorted `(type_id, sub_key)` pairs, one per
    neighbour, each further expanded one hop. Siblings produced by one call
    are always homogeneous (either all bare ints or all `(type_id, sub_key)`
    tuples), so `sorted()` never compares an int against a tuple.
    """
    neighbours = _neighbor_atoms_excluding(node, parent, bond_topology, bond_enum)
    if remaining <= 1:
        return tuple(sorted(atom_types[neighbours].tolist()))
    return tuple(
        sorted(
            (
                int(atom_types[nb]),
                _build_structure_key(
                    nb, node, remaining - 1, atom_types, bond_topology, bond_enum
                ),
            )
            for nb in neighbours
        )
    )


def _structure_key_to_string(key, type_to_name, remaining):
    """
    Convert a nested structure key (see :func:`_build_structure_key`) into
    a human-readable identifier fragment, e.g. ``"Si(O+O)+H()"``.
    """
    if remaining <= 1:
        return "+".join(type_to_name[t] for t in key)
    return "+".join(
        type_to_name[t]
        + "("
        + _structure_key_to_string(sub, type_to_name, remaining - 1)
        + ")"
        for t, sub in key
    )


@Substitution(params=_SAMPLER_INIT_PARAMS, name_out=_NAME_OUT_PARAM)
class MoleculeStructureSampler(Sampler):
    """
    Sampler class for molecular structure statistics.

    Parameters
    ----------
    %(name_out)s
    %(params)s
    steps : int, optional
        Number of bonding "steps" from the central atom to search for. Default
        is 1 (only directly bonded neighbours, e.g. "O(Si+Si)"). Step 2 would
        include information about the Si neighbours' in that example, e.g.
        "O(Si(O+O+O)+Si(O+O+O))", and so on for larger values.
    """

    def __init__(
        self,
        name_out: str,
        dimension: str,
        region,
        process_id: int,
        atom_lib: dict,
        masses: dict,
        num_frames: int,
        box: np.ndarray,
        system_properties: dict,
        steps: int = 1,
    ):
        valid_dimensions = ["MoleculeStructure"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(
                f"MoleculeStructureSampler does not support dimension {dimension}"
            )
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(
                "MoleculeStructureSampler requires a positive integer 'steps' "
                "parameter."
            )
        super().__init__(
            name_out,
            dimension,
            region,
            process_id,
            atom_lib,
            masses,
            num_frames,
            box,
            system_properties,
        )
        self._steps = steps
        self._input.update({"steps": steps})

        # Setup data
        self._data["num_frames"] = 0
        self._data["structure_counts"] = {}
        for atom_type in atom_lib.values():
            self._data["structure_counts"][atom_type] = {}

    def sample(
        self,
        frame_id: int,
        molecule_mask: dict,
        molecule_bond_atoms: dict,
        bond_mask: dict,
        frame: object,
        bond_enum: object,
        positions_transformed: np.ndarray,
    ):
        atom_types = frame.particles.particle_types.array
        bond_topology = frame.particles.bonds.topology.array
        positions = frame.particles.positions.array
        position_mask = self._region(positions)
        for atom_type in self._data["structure_counts"]:
            atoms = np.where(atom_types == atom_type)[0]
            for atom in atoms:
                key = _build_structure_key(
                    atom, None, self._steps, atom_types, bond_topology, bond_enum
                )
                if position_mask[atom]:
                    if key not in self._data["structure_counts"][atom_type]:
                        self._data["structure_counts"][atom_type][key] = 0
                    self._data["structure_counts"][atom_type][key] += 1
        self._data["num_frames"] += 1

    def join_samplers(self, num_cores: int) -> None:
        if self._process_id != -1:
            return
        combined_data = {}
        num_frames = 0
        type_to_name = {v: k for k, v in self._atom_lib.items()}
        for proc_data in super()._iter_process_data(num_cores):
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
                            name = (
                                atom
                                + "("
                                + _structure_key_to_string(
                                    structure, type_to_name, self._steps
                                )
                                + ")"
                            )
                            if name not in combined_data[atom]:
                                combined_data[atom][name] = 0
                            combined_data[atom][name] += count
        for atom in combined_data:
            if atom != "input_params":
                combined_data[atom] = dict(
                    sorted(
                        combined_data[atom].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                for structure in combined_data[atom]:
                    combined_data[atom][structure] /= num_frames

        save_object(combined_data, self._name_out + ".obj")

    def _combine_identifier(self, identifier: str, data: dict) -> dict:
        return {}
