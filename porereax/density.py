"""
### Module for sampling atomic densities.

It provides:
1. DensitySampler: A class to sample atomic densities on specified atoms and their bonds with for multiple dimensions:
    * Cartesian1D: Samples the density histogram along a specified Cartesian direction for the whole simulation box.
    * Time: Samples the number of atoms (with given bonds) per frame.
2. Functions to plot the sampled density data:
    * plot_hist: Plots density histograms from sampled data.
    * plot_time: Plots density over time from sampled data.
"""


import numpy as np
from porereax.meta_sampler import BondSampler, AtomSampler
import porereax.utils as utils
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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
            Sampling dimension. Supported: "Cartesian1D", "Time".
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
        valid_dimensions = ["Cartesian1D", "Cartesian2D", "Time"]
        if not isinstance(dimension, str) or dimension not in valid_dimensions:
            raise ValueError(f"DensitySampler does not support dimension {dimension}")
        if not isinstance(num_bins, (int)) or num_bins <= 0:
            raise ValueError("DensitySampler requires a positive integer 'num_bins' parameter.")
        if not isinstance(conditions, dict):
            raise ValueError("DensitySampler requires a dictionary 'conditions' parameter.")
        if "Charge" in conditions:
            charge_cond = conditions["Charge"]
            if (not isinstance(charge_cond, (list, tuple)) or 
                    len(charge_cond) != 2 or
                    charge_cond[0] >= charge_cond[1]):
                raise ValueError("DensitySampler 'conditions' parameter 'Charge' must be a list or tuple of two numbers (min_charge, max_charge) with min < max.")
        if "Angle" in conditions:
            angle_cond = conditions["Angle"]
            if (not isinstance(angle_cond, (list, tuple)) or 
                    len(angle_cond) != 2 or
                    angle_cond[0] >= angle_cond[1]):
                raise ValueError("DensitySampler 'conditions' parameter 'Angle' must be a list or tuple of two numbers (min_angle, max_angle) with min < max.")
        self.num_bins = num_bins
        self.direction = direction
        self.conditions = conditions
        super().__init__(name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, num_bins=num_bins, direction=direction, conditions=conditions)

        # Setup data
        for identifier, bonds_info in self.molecules.items():
            if self.dimension == "Time":
                self.data[identifier] = {"densities": np.zeros(num_frames), "num_frames": 0, }
            elif self.dimension == "Cartesian1D":
                if self.direction not in ["x", "y", "z"]:
                    raise ValueError("DensitySampler with 'Cartesian1D' dimension requires 'direction' parameter to be one of 'x', 'y', or 'z'.")
                dir_index = {"x": 0, "y": 1, "z": 2}[self.direction]
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=(0.0, box[dir_index]))
                self.data[identifier] = {"hist": hist, "bin_edges": bin_edges, "direction": dir_index, "num_frames": 0}
                self.data[identifier]["direction"] = dir_index
            elif self.dimension == "Cartesian2D":
                if self.direction not in ["xy", "xz", "yz"]:
                    raise ValueError("DensitySampler with 'Cartesian2D' dimension requires 'direction' parameter to be one of 'xy', 'xz', or 'yz'.")
                dir_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[self.direction]
                hist, x_edges, y_edges = np.histogram2d([], [], bins=self.num_bins, range=[[0.0, box[dir_indices[0]]], [0.0, box[dir_indices[1]]]])
                self.data[identifier] = {"hist": hist, "x_edges": x_edges, "y_edges": y_edges, "direction": dir_indices, "num_frames": 0}
                self.data[identifier]["direction"] = dir_indices

    def sample(self, frame: int, positions: np.ndarray, mol_index: dict, charges: np.ndarray, mol_bonds: dict, types: np.ndarray):
        """
        Sample atomic densities for the given frame.

        Parameters
        ----------
        frame : int
            Current frame number.
        positions : np.ndarray
            Array of atomic positions.
        mol_index : dict
            Mapping of molecule identifiers to atom indices.
        charges : np.ndarray
            Array of atomic charges.
        """
        for identifier in self.molecules:
            atom_indices = mol_index[identifier]
            # Apply conditions
            if "Charge" in self.conditions:
                min_charge, max_charge = self.conditions["Charge"]
                atom_charges = charges[atom_indices]
                charge_mask = (atom_charges >= min_charge) & (atom_charges <= max_charge)
                atom_indices = atom_indices[charge_mask]
            if "Angle" in self.conditions:
                angles = self.__get_atom_angles(atom_indices, positions, mol_bonds[identifier], types)
                min_angle, max_angle = self.conditions["Angle"]
                angle_mask = (angles >= min_angle) & (angles <= max_angle)
                angle_mask = np.any(angle_mask, axis=1)
                atom_indices = atom_indices[angle_mask]
            atom_positions = positions[atom_indices]
            self.data[identifier]["num_frames"] += 1
            if self.dimension == "Time":
                self.data[identifier]["densities"][frame] = atom_indices.shape[0]
            elif self.dimension == "Cartesian1D":
                direction = self.data[identifier]["direction"]
                hist, _ = np.histogram(atom_positions[:, direction], bins=self.num_bins, range=(0.0, self.box[direction]))
                self.data[identifier]["hist"] += hist
            elif self.dimension == "Cartesian2D":
                dir_x, dir_y = self.data[identifier]["direction"]
                hist, _, _ = np.histogram2d(atom_positions[:, dir_x], atom_positions[:, dir_y], bins=self.num_bins, range=[[0.0, self.box[dir_x]], [0.0, self.box[dir_y]]])
                self.data[identifier]["hist"] += hist

    def __get_atom_angles(self, atom_indices: np.ndarray, positions: np.ndarray, bonded_atoms: np.ndarray, types: np.ndarray):
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
        types : np.ndarray
            Array of atomic types.

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
        combined_data = {}
        for identifier in data_list:
            if identifier == "input_params":
                combined_data["input_params"] = data_list["input_params"]
                continue
            combined_data[identifier] = {}
            num_frames = np.sum(data_list[identifier]["num_frames"])
            combined_data[identifier]["num_frames"] = num_frames
            if self.dimension == "Time":
                combined_data[identifier]["densities"] = np.concatenate(data_list[identifier]["densities"])
            elif self.dimension == "Cartesian1D":
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self.num_bins) # TODO check normalization
                combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0) # TODO fix std calculation
                combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
                combined_data[identifier]["direction"] = data_list[identifier]["direction"][0]
            elif self.dimension == "Cartesian2D":
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros((self.num_bins, self.num_bins)) # TODO check normalization
                combined_data[identifier]["hist_std"] = np.std(data_list[identifier]["hist"], axis=0) # TODO fix std calculation
                combined_data[identifier]["x_edges"] = data_list[identifier]["x_edges"][0]
                combined_data[identifier]["y_edges"] = data_list[identifier]["y_edges"][0]
                combined_data[identifier]["direction"] = data_list[identifier]["direction"][0]
        utils.save_object(combined_data, self.folder + "/combined.obj")


def plot_hist(link_data: str, axis: Axes | bool=True, identifiers = [], colors = [], std=False, mean=False, plot_kwargs = {}):
    fig, ax, data, identifiers, colors = utils.plot_setup(link_data, axis, identifiers, colors)

    if data["input_params"]["dimension"] != "Cartesian1D":
        print("Data dimension is not 'Cartesian1D'. Cannot plot histogram.")
        return
    for i, identifier in enumerate(identifiers):
        if identifier == "input_params":
            continue
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        bin_edges = data[identifier]["bin_edges"]
        hist = data[identifier]["hist"]
        std_hist = data[identifier]["std_hist"] if std else None
        utils.plot_hist(ax, identifier, bin_edges, hist, colors[i % len(colors)], {}, std_hist)

    ax.set_xlabel(f"{data['input_params']['direction']} Position / nm")
    ax.set_ylabel("Density / atoms")

def plot_time(link_data: str, axis: Axes | bool=True, identifiers = [], colors = [], dt=50):
    fig, ax, data, identifiers, colors = utils.plot_setup(link_data, axis, identifiers, colors)
    if data["input_params"]["dimension"] != "Time":
        return
    for i, identifier in enumerate(identifiers):
        if identifier == "input_params":
            continue
        if identifier not in data:
            print(f"Warning: Identifier {identifier} not found in data.")
            continue
        time_data = data[identifier]
        time_points = np.arange(0, time_data["num_frames"] * dt, dt) / 1000  # Convert to ps
        density_data = time_data["densities"]
        color = colors[i % len(colors)] if colors else None
        ax.plot(time_points, density_data, label=identifier, color=color)
    ax.set_xlabel("Time / ps")
    ax.set_ylabel("Density / atoms")

def plot_2d_hist(link_data: str, identifier: str, transpose: bool=False):
    """
    Plot 2D density histogram from sampled data.

    Parameters
    ----------
    link_data : str
        Path to the data file containing sampled density data.
    identifier : str
        Molecule identifier to plot.
    """
    data = utils.load_object(link_data)
    if data["input_params"]["dimension"] != "Cartesian2D":
        return
    if identifier not in data:
        print(f"Warning: Identifier {identifier} not found in data.")
        return
    density_data = data[identifier]
    x_edges = density_data["x_edges"] / 10  # Convert to nm
    y_edges = density_data["y_edges"] / 10  # Convert to nm
    hist = density_data["hist"]

    X, Y = np.meshgrid(x_edges, y_edges)
    if transpose:
        X, Y = Y, X
    plt.figure()
    plt.pcolormesh(X, Y, hist.T, shading='auto')
    if transpose:
        plt.xlabel(f"{['x','y','z'][density_data['direction'][1]]} Position / nm")
        plt.ylabel(f"{['x','y','z'][density_data['direction'][0]]} Position / nm")
    else:
        plt.xlabel(f"{['x','y','z'][density_data['direction'][0]]} Position / nm")
        plt.ylabel(f"{['x','y','z'][density_data['direction'][1]]} Position / nm")
    plt.axis('scaled')
    plt.colorbar(label='Density / atoms')
    plt.show()