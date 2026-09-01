"""
Module for sampling bond angles between bonded atoms.

The module provides :class:`AngleSampler` to sample bond angle histograms for specified atom structures.
It supports filtering by specific A-B-C triplets or sampling all angles formed by the central atom B.
"""

import numpy as np
import porereax.utils as utils

from porereax.meta_sampler import AtomSampler, _ATOM_SAMPLER_INIT_PARAMS
from porereax.utils import Substitution


@Substitution(params=_ATOM_SAMPLER_INIT_PARAMS)
class AngleSampler(AtomSampler):
    """
    Sampler class for angles formed by three atoms.

    Parameters
    ----------
    %(params)s
    num_bins : int
        Number of bins for histogram sampling.
    angle : str
        Angle of interested atoms. Supported: "all", "A-B-C" where A, B, C are atom identifiers.
    """
    def __init__(self, name_out: str, atoms: list, dimension: str, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, num_bins: int, angle: str):
        valid_dimensions = ["Histogram"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"AngleSampler does not support dimension {dimension}")
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("AngleSampler requires a positive integer 'num_bins' parameter.")
        if not isinstance(angle, str):
            raise ValueError("AngleSampler requires 'angle' parameter to be a string.")
        if angle != "all":
            if len(angle.split("-")) != 3:
                raise ValueError("AngleSampler 'angle' parameter must be 'all' or in the format 'A-B-C'.")
            angle_atoms = angle.split("-")
            for atom in angle_atoms:
                if atom not in atom_lib:
                    raise ValueError(f"AngleSampler 'angle' parameter contains unknown atom identifier '{atom}'.")
            angle_list = [atom_lib[atom] for atom in angle_atoms]
        else:
            angle_list = []
        self._angle = angle_list
        self._num_bins = num_bins
        self._range = (0, 180)
        super().__init__(name_out, atoms, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        self._input.update({"num_bins": num_bins, "range": self._range, "angle": angle_list})

        # Remove atomstructures with less than 3 atoms or not matching A-B-C if specified
        molecules_to_remove = []
        for identifier, atoms_info in self._molecules.items():
            if self._angle and atoms_info["atom"] != self._angle[1]:
                molecules_to_remove.append(identifier)
            elif atoms_info["bonds"] is None or len(atoms_info["bonds"][0]) < 2:
                molecules_to_remove.append(identifier)
        for identifier in molecules_to_remove:
            del self._molecules[identifier]

        # Setup data
        for identifier, atoms_info in self._molecules.items():
            hist, bin_edges = np.histogram([], bins=self._num_bins, range=self._range)
            self._data[identifier] = {
                "num_frames": 0, 
                "num_angles": 0, 
                "mean_angle": 0.0, 
                "hist": hist, 
                "bin_edges": bin_edges, 
            }

    def sample(self, frame_id: int, molecule_mask: dict, molecule_bond_atoms: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        atom_types = frame.particles.particle_types.array
        positions = frame.particles.positions.array
        position_mask = self._region(positions)
        for identifier, bonds_info in self._molecules.items():
            mol_mask = position_mask & molecule_mask[identifier]
            atom_indices = np.where(mol_mask)[0]
            bonded_atoms = molecule_bond_atoms[identifier][atom_indices]
            if self._angle:
                atom_a_type = self._angle[0]
                atom_c_type = self._angle[2]
                bonded_types = atom_types[bonded_atoms]
            angles = []
            for i in range(bonded_atoms.shape[1]):
                for j in range(bonded_atoms.shape[1]):
                    if i == j:
                        continue
                    atom_a = bonded_atoms[:, i]
                    atom_b = atom_indices
                    atom_c = bonded_atoms[:, j]
                    if self._angle:
                        mask_a = bonded_types[:, i] == atom_a_type
                        mask_c = bonded_types[:, j] == atom_c_type
                        valid_mask = mask_a & mask_c
                        atom_a = atom_a[valid_mask]
                        atom_b = atom_b[valid_mask]
                        atom_c = atom_c[valid_mask]
                    vec_ab = utils.min_image_convention(positions[atom_a] - positions[atom_b], self._box)
                    vec_cb = utils.min_image_convention(positions[atom_c] - positions[atom_b], self._box)
                    cos_angle = np.sum(vec_ab * vec_cb, axis=1) / (np.linalg.norm(vec_ab, axis=1) * np.linalg.norm(vec_cb, axis=1))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    angles.extend(angle_deg.tolist())
            if angles:
                self._data[identifier]["num_frames"] += 1
                self._data[identifier]["num_angles"] += len(angles)
                self._data[identifier]["mean_angle"] += np.sum(angles)
                if self._dimension == "Histogram":
                    hist, _ = np.histogram(angles, bins=self._num_bins, range=self._range)
                    self._data[identifier]["hist"] += hist

    def _combine_identifier(self, identifier: str, data: dict) -> dict:
        num_frames = np.sum(data["num_frames"])
        num_angles = np.sum(data["num_angles"])
        mean = np.sum(data["mean_angle"]) / num_angles if num_angles > 0 else np.nan
        hist = np.sum(data["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self._num_bins) # TODO check normalization
        hist_std = np.std(data["hist"]) # TODO: fix std calculation
        mean_std = 0 # TODO: fix std calculation
        bin_edges = data["bin_edges"][0]
        return {
            "num_frames": num_frames,
            "num_angles": num_angles,
            "mean": mean,
            "hist": hist,
            "hist_std": hist_std,
            "mean_std": mean_std,
            "bin_edges": bin_edges,
        }
