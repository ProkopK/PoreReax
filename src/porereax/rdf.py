"""
Module for sampling radial distribution functions (RDF).

The module provides :class:`RdfSampler` for pair-distribution sampling of specified atom pairs.
"""


import numpy as np
import porereax.utils as utils

from porereax.meta_sampler import AtomSampler, _build_mol_dictionary, _validate_double_atoms, _SAMPLER_INIT_PARAMS, _NAME_OUT_PARAM
from porereax.utils import Substitution


@Substitution(params=_SAMPLER_INIT_PARAMS, name_out=_NAME_OUT_PARAM)
class RdfSampler(AtomSampler):
    """
    Sampler class for radial distribution functions (RDF).

    Parameters
    ----------
    %(name_out)s
    pairs : list
        List of atom pairs to sample, each specified as a list or tuple of two dictionaries:
        - Each dictionary should have keys: "atom" (str), "bonds" (list, optional)
    %(params)s
    num_bins : int
        Number of bins for histogram sampling.
    r_max : float
        Maximum distance for RDF calculation.
    """
    def __init__(self, name_out: str, pairs: list, dimension: str, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, num_bins: int, r_max: float):
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("RdfSampler requires a positive integer 'num_bins' parameter.")
        if not isinstance(r_max, (float, int)) or r_max <= 0:
            raise ValueError("RdfSampler requires a positive 'r_max' parameter.")

        self._num_bins = num_bins
        self._r_max = r_max

        # Extract atoms from pairs and validate format
        _validate_double_atoms(pairs, "RdfSampler", "pairs", allow_none=False)
        atoms = []
        for pair in pairs:
            atom1, atom2 = pair
            atoms.append(atom1)
            atoms.append(atom2)

        super().__init__(name_out, atoms, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        self._input.update({
            "num_bins": num_bins,
            "r_max": r_max,
        })

        # Build pair identifiers and setup data structures for each pair
        self._pairs = {}
        for pair in pairs:
            pair_A, pair_B = pair
            identifier_A = _build_mol_dictionary(pair_A["atom"], pair_A.get("bonds", None), atom_lib, "RDF Sampler")[0]
            identifier_B = _build_mol_dictionary(pair_B["atom"], pair_B.get("bonds", None), atom_lib, "RDF Sampler")[0]
            pair_key = f"{identifier_A}-{identifier_B}"
            self._pairs[pair_key] = (identifier_A, identifier_B)

            hist, bin_edges = np.histogram([], bins=self._num_bins, range=(0, self._r_max))
            self._data[pair_key] = {
                "num_frames": 0,
                "num_atoms_A": 0, # needed for normalization
                "num_atoms_B": 0, # needed for normalization
                "hist": hist,
                "bin_edges": bin_edges,
            }
        self._input["pairs"] = self._pairs


    def sample(self, frame_id: int, molecule_mask: dict, molecule_bond_atoms: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        from ovito.data import CutoffNeighborFinder

        # Create CutoffNeighborFinder for efficient neighbor search
        finder = CutoffNeighborFinder(self._r_max, frame)

        positions = frame.particles.positions.array
        position_mask = self._region(positions)

        for pair_key, (identifier_A, identifier_B) in self._pairs.items():
            # Get atom indices for both types
            atom_mask_A = molecule_mask[identifier_A] & position_mask
            atom_mask_B = molecule_mask[identifier_B] & position_mask
            atom_indices_A = np.where(atom_mask_A)[0]
            atom_indices_B = np.where(atom_mask_B)[0]

            pairs, pair_vectors = finder.find_all(atom_indices_A)

            mask = np.isin(pairs[:, 1], atom_indices_B)
            filtered_vectors = pair_vectors[mask]
            distances = np.linalg.norm(filtered_vectors, axis=1)

            hist, _ = np.histogram(distances, bins=self._num_bins, range=(0, self._r_max))
            self._data[pair_key]["hist"] += hist
            self._data[pair_key]["num_frames"] += 1
            self._data[pair_key]["num_atoms_A"] += atom_indices_A.size
            self._data[pair_key]["num_atoms_B"] += atom_indices_B.size

    def _combine_identifier(self, identifier: str, data: dict) -> dict:
        combined = {}
        num_frames = np.sum(data["num_frames"])
        num_atoms_A = np.sum(data["num_atoms_A"])
        num_atoms_B = np.sum(data["num_atoms_B"])
        combined["num_frames"] = num_frames
        combined["num_atoms_A"] = num_atoms_A
        combined["num_atoms_B"] = num_atoms_B

        # Sum histograms and normalize
        hist_sum = np.sum(data["hist"], axis=0)

        bin_edges = data["bin_edges"][0]

        # Calculate average number of atoms per frame
        avg_atoms_A = num_atoms_A / num_frames if num_frames > 0 else 0
        avg_atoms_B = num_atoms_B / num_frames if num_frames > 0 else 0

        # Calculate box volume
        box_volume = np.prod(self._box)

        # Calculate shell volumes: V = 4/3 * pi * (r_outer^3 - r_inner^3)
        r_inner = bin_edges[:-1]
        r_outer = bin_edges[1:]
        shell_volumes = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

        # Avoid division by zero
        shell_volumes = np.where(shell_volumes > 0, shell_volumes, 1e-10)

        # Normalize: g(r) = histogram / (N_frames * N_atoms_A * rho_B * V_shell)
        # This gives g(r) -> 1 for large r in a homogeneous system
        if num_frames > 0 and avg_atoms_A > 0 and avg_atoms_B > 0:
            combined["hist"] = box_volume * hist_sum / (num_frames * avg_atoms_A * avg_atoms_B * shell_volumes)
        else:
            combined["hist"] = np.zeros(self._num_bins)

        combined["hist_raw"] = hist_sum / num_frames if num_frames > 0 else np.zeros(self._num_bins)
        combined["hist_std"] = np.std(data["hist"], axis=0)
        combined["bin_edges"] = bin_edges
        return combined
