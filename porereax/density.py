"""
### Module for sampling atomic and bond densities.

It provides:
1. DensitySampler: A class to sample atomic densities on specified atoms and their bonds with for multiple dimensions:
    * Cartesian1D: Samples the density histogram along a specified Cartesian direction for the whole simulation box.
    * Cartesian2D: Samples the density histogram in a specified plane for the whole simulation box.
    * Time: Samples the number of atoms (with given bonds) per frame.
2. BondDensitySampler: A class to sample bond densities on specified bonds with for multiple dimensions:
    * Cartesian1D: Samples the density histogram along a specified Cartesian direction for the whole simulation box.
    * Cartesian2D: Samples the density histogram in a specified plane for the whole simulation box.
    * Time: Samples the number of bonds per frame.
"""


import numpy as np
from porereax.meta_sampler import BondSampler, AtomSampler
import porereax.utils as utils


def _validate_dimension(dimension: str, sampler_name: str):
    """Validate the dimension parameter."""
    valid_dimensions = ["Cartesian1D", "Cartesian2D", "Time"]
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

def _setup_data_structure(dimension: str, direction: str, num_frames: int, num_bins: int, box: np.ndarray, sampler_name: str):
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

def _record_density(data: dict, dimension: str, positions: np.ndarray, frame: int, num_bins: int, box: np.ndarray):
    """
    Record density data for the current frame.
    
    Parameters
    ----------
    data : dict
        Data structure for this identifier.
    dimension : str
        Sampling dimension.
    positions : np.ndarray
        Positions to record (Nx3 array).
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
        hist, _ = np.histogram(positions[:, direction], bins=num_bins, range=(0.0, box[direction]))
        data["hist"] += hist
    elif dimension == "Cartesian2D":
        dir_x, dir_y = data["direction"]
        hist, _, _ = np.histogram2d(positions[:, dir_x], positions[:, dir_y], bins=num_bins, range=[[0.0, box[dir_x]], [0.0, box[dir_y]]])
        data["hist"] += hist

def _join_data(data_list: dict, dimension: str, num_bins: int):
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
    for identifier in data_list:
        if identifier == "input_params":
            combined_data["input_params"] = data_list["input_params"]
            continue
        combined_data[identifier] = {}
        num_frames = np.sum(data_list[identifier]["num_frames"])
        combined_data[identifier]["num_frames"] = num_frames
        
        if dimension == "Time":
            combined_data[identifier]["densities"] = np.concatenate(data_list[identifier]["densities"])
        elif dimension == "Cartesian1D":
            combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(num_bins)
            combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0)
            combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
            combined_data[identifier]["direction"] = data_list[identifier]["direction"][0]
        elif dimension == "Cartesian2D":
            combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros((num_bins, num_bins))
            combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0)
            combined_data[identifier]["x_edges"] = data_list[identifier]["x_edges"][0]
            combined_data[identifier]["y_edges"] = data_list[identifier]["y_edges"][0]
            combined_data[identifier]["direction"] = data_list[identifier]["direction"][0]
    return combined_data


class DensitySampler(AtomSampler):
    """
    Sampler class for atomic densities.
    """
    def __init__(self, name_out: str, dimension: str, atoms: list, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, direction: str, conditions: dict = {}):
        """
        Sampler for atomic densities.

        Parameters
        ----------
        name_out : str
            Output folder name.
        dimension : str
            Sampling dimension. Supported: "Cartesian1D", "Cartesian2D", "Time".
        atoms : dict
            Dictionary defining atoms to sample.
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
            - "Charge": tuple (min_charge, max_charge)
            - "Angle": tuple (min_angle, max_angle) using angle type all
        """
        # Validate parameters using helper
        _validate_dimension(dimension, "DensitySampler")
        _validate_num_bins(num_bins, "DensitySampler")
        _validate_conditions(conditions, "DensitySampler")
        _validate_condition_range(conditions, "Charge", "DensitySampler")
        _validate_condition_range(conditions, "Angle", "DensitySampler")
        
        self.num_bins = num_bins
        self.direction = direction
        self.conditions = conditions
        super().__init__(name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, num_bins=num_bins, direction=direction, conditions=conditions)

        # Setup data using helper
        for identifier in self.molecules:
            self.data[identifier] = _setup_data_structure(
                self.dimension, self.direction, num_frames, self.num_bins, box, "DensitySampler"
            )

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_index: dict, frame: object, bond_enum: object):
        charges = frame.particles.get("Charge").array if "Charge" in frame.particles else np.zeros(frame.particles.count)
        positions = frame.particles.positions.array

        for identifier in self.molecules:
            atom_indices = mol_index[identifier]
            # Apply conditions
            if "Charge" in self.conditions:
                min_charge, max_charge = self.conditions["Charge"]
                atom_charges = charges[atom_indices]
                charge_mask = (atom_charges >= min_charge) & (atom_charges <= max_charge)
                atom_indices = atom_indices[charge_mask]
            if "Angle" in self.conditions:
                angles = self.__get_atom_angles(atom_indices, positions, mol_bonds[identifier])
                min_angle, max_angle = self.conditions["Angle"]
                angle_mask = (angles >= min_angle) & (angles <= max_angle)
                angle_mask = np.any(angle_mask, axis=1)
                atom_indices = atom_indices[angle_mask]
            
            atom_positions = positions[atom_indices]
            # Record density using helper
            _record_density(
                self.data[identifier], self.dimension, atom_positions, frame_id, self.num_bins, self.box
            )

    def __get_atom_angles(self, atom_indices: np.ndarray, positions: np.ndarray, bonded_atoms: np.ndarray):
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
        angles = np.zeros((bonded_atoms.shape[0], bonded_atoms.shape[1] * bonded_atoms.shape[1]- bonded_atoms.shape[1]))
        for i in range(bonded_atoms.shape[1]):
            for j in range(bonded_atoms.shape[1]):
                if i == j:
                    continue
                atom_a = bonded_atoms[:, i]
                atom_b = atom_indices
                atom_c = bonded_atoms[:, j]
                vec_ab = utils.min_image_convention(positions[atom_a] - positions[atom_b], self.box)
                vec_cb = utils.min_image_convention(positions[atom_c] - positions[atom_b], self.box)
                cos_angle = np.sum(vec_ab * vec_cb, axis=1) / (np.linalg.norm(vec_ab, axis=1) * np.linalg.norm(vec_cb, axis=1))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                angles[:, i * (bonded_atoms.shape[1]-1) + j - (1 if j > i else 0)] = angle_deg
        return np.array(angles)

    def join_samplers(self, num_cores):
        """
        Join data from multiple samplers after parallel processing.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used.
        """
        data_list = super().join_samplers(num_cores)
        # Use helper to join data
        combined_data = _join_data(data_list, self.dimension, self.num_bins)
        utils.save_object(combined_data, self.name_out + ".obj")


class BondDensitySampler(BondSampler):
    """
    Sampler class for bond densities.
    """
    def __init__(self, name_out: str, dimension: str, bonds: list, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, direction: str, conditions: dict = {}):
        """
        Sampler for bond densities.

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
        # Validate parameters using helper
        _validate_dimension(dimension, "BondDensitySampler")
        _validate_num_bins(num_bins, "BondDensitySampler")
        _validate_conditions(conditions, "BondDensitySampler")
        _validate_condition_range(conditions, "Bond Length", "BondDensitySampler")
        
        self.num_bins = num_bins
        self.direction = direction
        self.conditions = conditions
        super().__init__(name_out, dimension, bonds, process_id, atom_lib, masses, num_frames, box, num_bins=num_bins, direction=direction, conditions=conditions)

        # Setup data using helper
        for identifier in self.bonds:
            self.data[identifier] = _setup_data_structure(
                self.dimension, self.direction, num_frames, self.num_bins, box, "BondDensitySampler"
            )

    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_index: dict, frame: object, bond_enum: object):
        bond_topology = frame.particles.bonds.topology.array
        bond_periodic_images = frame.particles.bonds.pbc_vectors.array
        positions = frame.particles.positions.array
        
        for identifier in self.bonds:
            bond_indices = bond_index[identifier]
            if bond_indices.size == 0:
                continue
            
            bonds = bond_topology[bond_indices]
            bond_positions = positions[bonds]
            
            # Calculate bond midpoints
            bond_midpoints = (bond_positions[:, 0, :] + bond_positions[:, 1, :]) / 2.0
            periodic_shifts = bond_periodic_images[bond_indices] * self.box
            bond_midpoints += periodic_shifts / 2.0 # TODO: Check if this is correct
            
            # Apply Bond Length condition if specified
            if "Bond Length" in self.conditions:
                min_length, max_length = self.conditions["Bond Length"]
                bond_vectors = utils.min_image_convention(bond_positions[:, 0, :] - bond_positions[:, 1, :], self.box)
                bond_lengths = np.linalg.norm(bond_vectors, axis=1)
                length_mask = (bond_lengths >= min_length) & (bond_lengths <= max_length)
                bond_midpoints = bond_midpoints[length_mask]
            
            # Record density using helper
            _record_density(
                self.data[identifier], self.dimension, bond_midpoints, frame_id, self.num_bins, self.box
            )

    def join_samplers(self, num_cores):
        """
        Join data from multiple samplers after parallel processing.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used.
        """
        data_list = super().join_samplers(num_cores)
        # Use helper to join data
        combined_data = _join_data(data_list, self.dimension, self.num_bins)
        utils.save_object(combined_data, self.name_out + ".obj")
