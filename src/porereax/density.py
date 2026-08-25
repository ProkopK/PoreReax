"""
Module for sampling atomic and bond densities.

The module provides:

1. :class:`DensitySampler`: A class to sample atomic densities of specified atom structures.
2. :class:`BondDensitySampler`: A class to sample bond densities of specified bonds.
3. :class:`ReactionSampler`: A class to sample reaction events based on bond formation and breaking.

All samplers support multiple dimensions for density sampling:

- "Cartesian1D": Samples the density histogram along a specified Cartesian direction for the whole simulation box.
- "Cartesian2D": Samples the density histogram in a specified plane for the whole simulation box.
- "Time": Samples the number of atoms (with given bonds) or bonds per frame
- "Pore1D": Samples the density histogram along a specified direction in a cylindrical pore.
- "Pore2D": Samples the density histogram in a specified plane in a cylindrical pore.
"""


import numpy as np
from porereax.meta_sampler import BondSampler, AtomSampler, Sampler, _build_mol_dictionary, _validate_double_atoms, _ATOM_SAMPLER_INIT_PARAMS, _BOND_SAMPLER_INIT_PARAMS, _SAMPLER_INIT_PARAMS, _NAME_OUT_PARAM
import porereax.utils as utils
from scipy.sparse import coo_matrix
from typing import Literal
from porereax.utils import Substitution, Appender


_DIRECTION_PARAM = """
direction : str
    Direction options, depending on the dimension:

    - ("x", "y", or "z") for "Cartesian1D".
    - ("xy", "xz", or "yz") for "Cartesian2D".
    - ("r", "p", or "d") for "Pore1D".
    - ("rp", "rz", or "pz") for "Pore2D".
"""

type Dimension = Literal["Cartesian1D", "Cartesian2D", "Time", "Pore1D", "Pore2D"]


def _validate_dimension(dimension: Dimension, sampler_name: str):
    """Validate the dimension parameter."""
    valid_dimensions = {"Cartesian1D", "Cartesian2D", "Time", "Pore1D", "Pore2D"}
    if not isinstance(dimension, str) or dimension not in valid_dimensions:
        raise ValueError(f"{sampler_name} does not support dimension {dimension}")

def _validate_num_bins(num_bins: int, sampler_name: str):
    """Validate the num_bins parameter."""
    if not isinstance(num_bins, (int)) or num_bins <= 0:
        raise ValueError(f"{sampler_name} requires a positive integer 'num_bins' parameter.")

def _validate_conditions(conditions: dict, sampler_name: str):
    """Validate the conditions parameter."""
    if not isinstance(conditions, dict):
        raise ValueError(f"{sampler_name} requires a dictionary 'conditions' parameter.")

def _validate_condition_range(conditions: dict, condition_name: str, sampler_name: str):
    """Validate a specific condition range (Charge, Angle, Bond Length)."""
    if condition_name in conditions:
        cond = conditions[condition_name]
        if (not isinstance(cond, (list, tuple)) or
                len(cond) != 2 or
                cond[0] >= cond[1]):
            raise ValueError(f"{sampler_name} 'conditions' parameter '{condition_name}' must be a list or tuple of two numbers (min, max) with min < max.")

def _setup_data_structure(dimension: Dimension, direction: str, num_frames: int, num_bins: int, box: np.ndarray, sampler_name: str, system_properties: dict | None):
    """
    Setup the data structure for a given dimension.

    Returns
    -------
    dict
        Data structure with initialized arrays and metadata.
    """
    if dimension == "Time":
        return {"densities": np.zeros(num_frames), "num_frames": 0}
    elif dimension == "Cartesian1D":
        if direction not in ["x", "y", "z"]:
            raise ValueError(f"{sampler_name} with 'Cartesian1D' dimension requires 'direction' parameter to be one of 'x', 'y', or 'z'.")
        dir_index = {"x": 0, "y": 1, "z": 2}[direction]
        hist, bin_edges = np.histogram([], bins=num_bins, range=(0.0, box[dir_index]))
        return {"hist": hist, "bin_edges": bin_edges, "direction": dir_index, "num_frames": 0}
    elif dimension == "Cartesian2D":
        if direction not in ["xy", "xz", "yz"]:
            raise ValueError(f"{sampler_name} with 'Cartesian2D' dimension requires 'direction' parameter to be one of 'xy', 'xz', or 'yz'.")
        dir_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[direction]
        hist, x_edges, y_edges = np.histogram2d([], [], bins=num_bins, range=[[0.0, box[dir_indices[0]]], [0.0, box[dir_indices[1]]]])
        return {"hist": hist, "x_edges": x_edges, "y_edges": y_edges, "direction": dir_indices, "num_frames": 0}
    elif dimension.startswith("Pore"):
        if system_properties is None:
            raise ValueError(f"{sampler_name} with 'Pore' dimension requires a given system")
        elif system_properties["type"] == "cylinder":
            center = system_properties["center"]
            max_r = np.min([center[0], center[1], box[0] - center[0], box[1] - center[1]])
            r2_edges = np.linspace(0.0, max_r**2, num_bins + 1)
            r_edges = np.sqrt(r2_edges)
            p_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
            d_edges = np.linspace(-center[2]/2, center[2]/2, num_bins + 1)
            z_edges = np.linspace(0.0, box[2], num_bins + 1)
            if direction in ["r", "p", "d"] and dimension == "Pore1D":
                dir_index = {"r": 3, "p": 4, "d": 6}[direction]
                if direction == "r":
                    bin_edges = r_edges
                elif direction == "p":
                    bin_edges = p_edges
                else:  # direction == "d"
                    bin_edges = d_edges
                hist, _ = np.histogram([], bins=num_bins)
                return {"hist": hist, "bin_edges": bin_edges, "direction": dir_index, "num_frames": 0}
            elif direction in ["rp", "rz", "pz"] and dimension == "Pore2D":
                dir_indices = {"rp": (3, 4), "rz": (3, 5), "pz": (4, 5)}[direction]
                if direction == "rp":
                    x_edges = r_edges
                    y_edges = p_edges
                elif direction == "rz":
                    x_edges = r_edges
                    y_edges = z_edges
                else:  # direction == "pz"
                    x_edges = p_edges
                    y_edges = z_edges
                hist, _, _ = np.histogram2d([], [], bins=num_bins)
                return {"hist": hist, "x_edges": x_edges, "y_edges": y_edges, "direction": dir_indices, "num_frames": 0}

def _record_density(data: dict, dimension: Dimension, positions: np.ndarray, frame: int):
    """
    Record density data for the current frame.

    Parameters
    ----------
    data : dict
        Data structure for this identifier.
    dimension : str
        Sampling dimension.
    positions : np.ndarray
        Positions to record (Nx3 array) or (Nx4 array) for Pore dimensions.
    frame : int
        Current frame number.
    num_bins : int
        Number of bins for histogramming.
    box : np.ndarray
        Simulation box dimensions.
    """
    data["num_frames"] += 1

    if dimension == "Time":
        data["densities"][frame] = positions.shape[0]
    elif dimension == "Cartesian1D":
        direction = data["direction"]
        hist, _ = np.histogram(positions[:, direction], bins=data["bin_edges"])
        data["hist"] += hist
    elif dimension == "Cartesian2D":
        dir_x, dir_y = data["direction"]
        hist, _, _ = np.histogram2d(positions[:, dir_x], positions[:, dir_y], bins=[data["x_edges"], data["y_edges"]])
        data["hist"] += hist
    elif dimension == "Pore1D":
        direction = data["direction"] - 3
        hist, _ = np.histogram(positions[:, direction], bins=data["bin_edges"])
        data["hist"] += hist
    elif dimension == "Pore2D":
        dir_x, dir_y = (data["direction"][0] - 3, data["direction"][1] - 3)
        hist, _, _ = np.histogram2d(positions[:, dir_x], positions[:, dir_y], bins=[data["x_edges"], data["y_edges"]])
        data["hist"] += hist

def _join_data(data_list: dict, dimension: Dimension, num_bins: int):
    """
    Join data from multiple samplers after parallel processing.

    Parameters
    ----------
    data_list : dict
        Dictionary containing lists of data from each process.
    dimension : str
        Sampling dimension.
    num_bins : int
        Number of bins.

    Returns
    -------
    dict
        Combined data structure.
    """
    combined_data = {}
    input_params = data_list.pop("input_params", None)
    combined_data["input_params"] = input_params
    for identifier in data_list:
        combined_data[identifier] = {}
        num_frames = np.sum(data_list[identifier]["num_frames"])
        combined_data[identifier]["num_frames"] = num_frames

        if dimension == "Time":
            combined_data[identifier]["densities"] = np.concatenate(data_list[identifier]["densities"])
        elif dimension == "Cartesian1D" or dimension == "Pore1D":
            combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(num_bins)
            combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0)
            combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
            combined_data[identifier]["direction"] = data_list[identifier]["direction"][0]
        elif dimension == "Cartesian2D" or dimension == "Pore2D":
            combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros((num_bins, num_bins))
            combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0)
            combined_data[identifier]["x_edges"] = data_list[identifier]["x_edges"][0]
            combined_data[identifier]["y_edges"] = data_list[identifier]["y_edges"][0]
            combined_data[identifier]["direction"] = data_list[identifier]["direction"][0]
    return combined_data


@Substitution(params=_ATOM_SAMPLER_INIT_PARAMS, direction=_DIRECTION_PARAM)
class DensitySampler(AtomSampler):
    """
    Sampler class for atomic densities.

    Parameters
    ----------
    %(params)s
    num_bins : int
        Number of bins for histogram sampling.
    %(direction)s
    conditions : dict, optional
        Additional conditions for sampling.
        - "Charge": tuple (min_charge, max_charge)
        - "Angle": tuple (min_angle, max_angle) using angle type all
    """
    def __init__(self, name_out: str, atoms: list, dimension: Dimension, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, num_bins: int, direction: str, conditions: dict = {}):
        # Validate parameters
        _validate_dimension(dimension, "DensitySampler")
        _validate_num_bins(num_bins, "DensitySampler")
        _validate_conditions(conditions, "DensitySampler")
        _validate_condition_range(conditions, "Charge", "DensitySampler")
        _validate_condition_range(conditions, "Angle", "DensitySampler")

        self._num_bins = num_bins
        self._direction = direction
        self._conditions = conditions
        super().__init__(name_out, atoms, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        self._input.update({
            "num_bins": num_bins,
            "direction": direction,
            "conditions": conditions,
        })

        # Setup data
        for identifier in self._molecules:
            self._data[identifier] = _setup_data_structure(
                self._dimension, self._direction, num_frames, self._num_bins, box, "DensitySampler", self._system_properties
            )

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        positions = frame.particles.positions.array
        position_mask = self._region(positions)
        for identifier in self._molecules:
            mol_mask = mol_index[identifier] & position_mask
            # Apply conditions
            if "Charge" in self._conditions:
                charges = frame.particles.get("Charge").array if "Charge" in frame.particles else np.zeros(frame.particles.count)
                min_charge, max_charge = self._conditions["Charge"]
                charge_mask = (charges >= min_charge) & (charges <= max_charge)
                mol_mask = mol_mask & charge_mask
            if "Angle" in self._conditions:
                # atom_indices = np.where(mol_mask)[0]
                atom_indices = np.arange(positions.shape[0])
                angles = self._get_atom_angles(atom_indices, positions, mol_bonds[identifier])
                min_angle, max_angle = self._conditions["Angle"]
                angle_mask = (angles >= min_angle) & (angles <= max_angle)
                angle_mask = np.any(angle_mask, axis=1)
                mol_mask = mol_mask & angle_mask

            if self._dimension in ("Pore1D", "Pore2D"):
                atom_positions = positions_transformed[mol_mask]
            else:
                atom_positions = positions[mol_mask]
            _record_density(
                self._data[identifier],
                self._dimension,
                atom_positions,
                frame_id,
            )

    def _get_atom_angles(self, atom_indices: np.ndarray, positions: np.ndarray, bonded_atoms: np.ndarray):
        """
        Calculate angles for atoms based on their bonded neighbors.

        Parameters
        ----------
        atom_indices : np.ndarray
            Indices of the central atoms.
        positions : np.ndarray
            Array of atomic positions.
        bonded_atoms : np.ndarray
            Array of bonded atom indices for each central atom.

        Returns
        -------
        angles : np.ndarray
            Calculated angles in degrees for the central atoms.
        """
        angles = np.zeros((bonded_atoms.shape[0], bonded_atoms.shape[1] * (bonded_atoms.shape[1] - 1)))
        for i in range(bonded_atoms.shape[1]):
            for j in range(bonded_atoms.shape[1]):
                if i == j:
                    continue
                atom_a = bonded_atoms[:, i]
                atom_b = atom_indices
                atom_c = bonded_atoms[:, j]
                vec_ab = utils.min_image_convention(positions[atom_a] - positions[atom_b], self._box)
                vec_cb = utils.min_image_convention(positions[atom_c] - positions[atom_b], self._box)
                cos_angle = np.sum(vec_ab * vec_cb, axis=1) / (np.linalg.norm(vec_ab, axis=1) * np.linalg.norm(vec_cb, axis=1))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                angles[:, i * (bonded_atoms.shape[1] - 1) + j - (1 if j > i else 0)] = angle_deg
        return np.array(angles)

    def join_samplers(self, num_cores: int) -> None:
        data_list = super()._collect_sampler_data(num_cores)
        combined_data = _join_data(data_list, self._dimension, self._num_bins)
        utils.save_object(combined_data, self._name_out + ".obj")


@Substitution(params=_BOND_SAMPLER_INIT_PARAMS, direction=_DIRECTION_PARAM)
class BondDensitySampler(BondSampler):
    """
    Sampler class for bond densities.

    Parameters
    ----------
    %(params)s
    num_bins : int
        Number of bins for histogram sampling.
    %(direction)s
    conditions : dict, optional
        Additional conditions for sampling.
        - "Bond Length": tuple (min_length, max_length)
    """
    def __init__(self, name_out: str, bonds: list, dimension: Dimension, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, num_bins: int, direction: str, conditions: dict = {}):
        """
        Sampler for bond densities.

        Parameters
        ----------
        name_out : str
            Output folder name.
        bonds : list
            List of bonds to sample, each specified as a dictionary with keys:
            - "bond": str, the bond in format "A-B"
            - "bonds_A": list, optional, list of bonded atom types for atom A
            - "bonds_B": list, optional, list of bonded atom types for atom B
        dimension : str
            Sampling dimension. Supported: "Cartesian1D", "Cartesian2D", "Time", "Pore1D", "Pore2D".
        region : callable

        Parameters
        ----------
        name_out : str
            Output folder name.
        dimension : str
            Sampling dimension. Supported: "Cartesian1D", "Cartesian2D", "Time".
        bonds : list
            List of bonds to sample, each specified as a dictionary with keys:
            - "bond": str, the bond in format "A-B"
            - "bonds_A": list, optional, list of bonded atom types for atom A
            - "bonds_B": list, optional, list of bonded atom types for atom B
        process_id : int
            Process ID for parallel sampling.
        atom_lib : dict
            Dictionary mapping atom type strings to their type IDs.
        masses : dict
            Dictionary mapping atom type strings to their masses.
        num_frames : int
            Total number of frames to sample.
        box : np.ndarray
            Simulation box dimensions.
        num_bins : int
            Number of bins for Cartesian sampling along each axis.
        direction : str
            Direction for Cartesian sampling. Options:
            - ("x", "y", or "z") for "Cartesian1D".
            - ("xy", "xz", or "yz") for "Cartesian2D".
        conditions : dict, optional
            Additional conditions for sampling.
            - "Bond Length": tuple (min_length, max_length)
        """
        # Validate parameters
        _validate_dimension(dimension, "BondDensitySampler")
        _validate_num_bins(num_bins, "BondDensitySampler")
        _validate_conditions(conditions, "BondDensitySampler")
        _validate_condition_range(conditions, "Bond Length", "BondDensitySampler")

        self._num_bins = num_bins
        self._direction = direction
        self._conditions = conditions
        super().__init__(name_out, bonds, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        self._input.update({
            "num_bins": num_bins,
            "direction": direction,
            "conditions": conditions,
        })

        # Setup data
        for identifier in self._bonds:
            self._data[identifier] = _setup_data_structure(
                self._dimension, self._direction, num_frames, self._num_bins, box, "BondDensitySampler", self._system_properties
            )

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        bond_topology = frame.particles.bonds.topology.array
        positions = frame.particles.positions.array

        for identifier in self._bonds:
            bond_indices = bond_mask[identifier]

            bonds = bond_topology[bond_indices]
            bond_positions = positions[bonds]

            # Calculate bond midpoints
            bond_midpoints = utils.min_image_midpoint(bond_positions[:, 0, :], bond_positions[:, 1, :], self._box)

            # Apply Bond Length condition if specified
            if "Bond Length" in self._conditions:
                min_length, max_length = self._conditions["Bond Length"]
                bond_vectors = utils.min_image_convention(bond_positions[:, 0, :] - bond_positions[:, 1, :], self._box)
                bond_lengths = np.linalg.norm(bond_vectors, axis=1)
                length_mask = (bond_lengths >= min_length) & (bond_lengths <= max_length)
                bond_midpoints = bond_midpoints[length_mask]

            # Record density
            _record_density(
                self._data[identifier],
                self._dimension,
                bond_midpoints,
                frame_id,
            )

    def join_samplers(self, num_cores: int) -> None:
        data_list = super()._collect_sampler_data(num_cores)
        combined_data = _join_data(data_list, self._dimension, self._num_bins)
        utils.save_object(combined_data, self._name_out + ".obj")


@Substitution(params=_SAMPLER_INIT_PARAMS, direction=_DIRECTION_PARAM, name_out=_NAME_OUT_PARAM)
class ReactionSampler(AtomSampler):
    """
    Sampler class for reaction events based on bond formation and breaking.

    Parameters
    ----------
    %(name_out)s
    reactions : list
        List of reactions to sample, each specified as a tuple of two dictionaries (reactant_dict, product_dict):
        Each dictionary should have keys:

        - "atom": str, the atom type string.
        - "bonds": list, optional, list of bonded atom types.
    %(params)s
    num_bins : int
        Number of bins for histogram sampling.
    %(direction)s
    position : str
        Position to sample for reaction events. Options:

        - "center": Sample at the midpoint of the reactant and product positions.
        - "reactant": Sample at the position of the reactant.
        - "product": Sample at the position of the product.
    """
    def __init__(self, name_out: str, reactions: list, dimension: Dimension, region, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, system_properties: dict, num_bins: int, direction: str, position: Literal["center", "reactant", "product"]):
        # Validate parameters
        _validate_dimension(dimension, "ReactionSampler")
        _validate_num_bins(num_bins, "ReactionSampler")
        if position not in ["center", "reactant", "product"]:
            raise ValueError(f"ReactionSampler requires 'position' parameter to be one of 'center', 'reactant', or 'product'.")

        self._num_bins = num_bins
        self._direction = direction
        self._position = position

        # Extract atoms from reactions and validate format
        _validate_double_atoms(reactions, "ReactionSampler", "reactions", allow_none=True)
        atoms = []
        for reaction in reactions:
            reactant, product = reaction
            if reactant is not None:
                atoms.append(reactant)
            if product is not None:
                atoms.append(product)

        super().__init__(name_out, atoms, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        self._input.update({
            "num_bins": num_bins,
            "direction": direction,
            "position": position,
        })

        # Build reaction identifiers and setup data structures for each reaction
        self._reactions = {}
        for reaction in reactions:
            reactant, product = reaction
            identifier_reactant = _build_mol_dictionary(reactant["atom"], reactant.get("bonds", None), atom_lib, "Reaction Sampler")[0] if reactant is not None else "X"
            identifier_product = _build_mol_dictionary(product["atom"], product.get("bonds", None), atom_lib, "Reaction Sampler")[0] if product is not None else "X"
            reaction_key = f"{identifier_reactant}-{identifier_product}"
            self._reactions[reaction_key] = (identifier_reactant, identifier_product)
            self._data[reaction_key] = _setup_data_structure(
                            self._dimension, self._direction, num_frames-1, self._num_bins, box, "ReactionSampler", self._system_properties
                        )
        self._input["reactions"] = self._reactions
        self._pre_positions = None
        self._cur_positions = None
        self._pre_mol_index = None
        self._cur_mol_index = None
        self._pre_bonds = None
        self._cur_bonds = None

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        cur_topology = frame.particles.bonds.topology.array

        self._pre_positions = self._cur_positions
        self._pre_mol_index = self._cur_mol_index
        self._pre_bonds = self._cur_bonds
        self._cur_positions = frame.particles.positions.array
        self._cur_mol_index = {key: np.copy(value) for key, value in mol_index.items()}
        self._cur_bonds = coo_matrix((np.ones(cur_topology.shape[0]), (cur_topology[:, 0], cur_topology[:, 1])), shape=(self._cur_positions.shape[0], self._cur_positions.shape[0]), dtype=bool)
        if self._pre_positions is None:
            return

        if self._position == "center":
            positions = utils.min_image_midpoint(self._pre_positions, self._cur_positions, self._box)
        elif self._position == "reactant":
            positions = self._pre_positions
        elif self._position == "product":
            positions = self._cur_positions
        position_mask = self._region(positions)

        reaction_events = (self._pre_bonds - self._cur_bonds).tocoo()
        reaction_indices = np.unique(np.concatenate((reaction_events.row, reaction_events.col)))

        for reaction_key, (identifier_reactant, identifier_product) in self._reactions.items():
            reactant_mask = self._pre_mol_index[identifier_reactant] if identifier_reactant != "X" else True
            product_mask = self._cur_mol_index[identifier_product] if identifier_product != "X" else True
            reaction_mask = reactant_mask & product_mask & position_mask
            reaction_key_indices = reaction_indices[reaction_mask[reaction_indices]]

            if self._dimension in ("Pore1D", "Pore2D"):
                reaction_positions = positions_transformed[reaction_key_indices]
            else:
                reaction_positions = positions[reaction_key_indices]
            _record_density(
                self._data[reaction_key],
                self._dimension,
                reaction_positions,
                frame_id - 1,
            )

    def join_samplers(self, num_cores: int) -> None:
        data_list = super()._collect_sampler_data(num_cores)
        combined_data = _join_data(data_list, self._dimension, self._num_bins)
        utils.save_object(combined_data, self._name_out + ".obj")
