import numpy as np
from porereax.meta_sampler import AtomSampler
import porereax.utils as utils
from matplotlib.axes import Axes


class AngleSampler(AtomSampler):
    """
    Sampler class for angles formed by three atoms.
    """
    def __init__(self, name_out: str, dimension: str, atoms: list, process_id: int, atom_lib: dict, masses: dict, num_frames: int, box: np.ndarray, num_bins: int, angle: str):
        """
        Sampler for angles formed by three atoms.

        Parameters
        ----------
        name_out : str
            Output folder name.
        dimension : str
            Sampling dimension. Currently only "Histogram" is supported.
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
            Number of bins for histogram sampling.
        angle : str
            Angle of interested atoms. Supported: "all", "A-B-C" where A, B, C are atom identifiers.
        """
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
            angle = [atom_lib[atom] for atom in angle_atoms]
        else:
            angle = []
        self.angle = angle
        self.num_bins = num_bins
        self.range = (0, 180)
        super().__init__(name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, num_bins=self.num_bins, range=self.range, angle=self.angle)

        # Setup data
        for identifier, atoms_info in self.molecules.items():
            if self.dimension == "Histogram":
                hist, bin_edges = np.histogram([], bins=self.num_bins, range=self.range)
                self.data[identifier] = {"num_frames": 0, "num_angles": 0, "mean_angle": 0.0, "hist": hist, "bin_edges": bin_edges, }

    def sample(self, frame: int, positions: np.ndarray, mol_index: dict, mol_bonds: dict, types: np.ndarray):
        """
        Sample angles for the given frame.

        Parameters
        ----------
        frame : int
            Current frame index.
        positions : np.ndarray
            Array of atomic positions.
        mol_index : dict
            Dictionary mapping molecule identifiers to atom indices.
        mol_bonds : dict
            Dictionary mapping molecule identifiers to all atoms the central atom is bonded to.
        types : np.ndarray
            Array of atomic types.
        """
        for identifier, bonds_info in self.molecules.items():
            atom_indices = mol_index[identifier]
            angles = []
            bonded_atoms = mol_bonds[identifier]
            if bonded_atoms.shape[1] < 2:
                continue
            if self.angle:
                atom_a_type = self.angle[0]
                atom_b_type = self.angle[1]
                atom_c_type = self.angle[2]
                if bonds_info["atom"] != atom_b_type:
                    continue
                bonded_types = types[bonded_atoms]
            for i in range(bonded_atoms.shape[1]):
                for j in range(bonded_atoms.shape[1]):
                    if i == j:
                        continue
                    atom_a = bonded_atoms[:, i]
                    atom_b = atom_indices
                    atom_c = bonded_atoms[:, j]
                    if self.angle:
                        mask_a = bonded_types[:, i] == atom_a_type
                        mask_c = bonded_types[:, j] == atom_c_type
                        valid_mask = mask_a & mask_c
                        if not np.any(valid_mask):
                            continue
                        atom_a = atom_a[valid_mask]
                        atom_b = atom_b[valid_mask]
                        atom_c = atom_c[valid_mask]
                    vec_ab = utils.min_image_convention(positions[atom_a] - positions[atom_b], self.box)
                    vec_cb = utils.min_image_convention(positions[atom_c] - positions[atom_b], self.box)
                    cos_angle = np.sum(vec_ab * vec_cb, axis=1) / (np.linalg.norm(vec_ab, axis=1) * np.linalg.norm(vec_cb, axis=1))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    angles.extend(angle_deg.tolist())
            if angles:
                self.data[identifier]["num_frames"] += 1
                self.data[identifier]["num_angles"] += len(angles)
                self.data[identifier]["mean_angle"] += np.sum(angles)
                if self.dimension == "Histogram":
                    hist, _ = np.histogram(angles, bins=self.num_bins, range=self.range)
                    self.data[identifier]["hist"] += hist

    def join_samplers(self, num_cores):
        """
        Join sampler data from multiple processes.

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
            if self.dimension == "Histogram":
                num_frames = np.sum(data_list[identifier]["num_frames"])
                num_angles = np.sum(data_list[identifier]["num_angles"])
                combined_data[identifier]["num_frames"] = num_frames
                combined_data[identifier]["num_angles"] = num_angles
                combined_data[identifier]["mean_angle"] = np.sum(data_list[identifier]["mean_angle"]) / num_angles if num_angles > 0 else np.nan
                combined_data[identifier]["hist"] = np.sum(data_list[identifier]["hist"], axis=0) / num_frames if num_frames > 0 else np.zeros(self.num_bins) # TODO check normalization
                combined_data[identifier]["std_angle"] = 0 # TODO: fix std calculation
                combined_data[identifier]["std_hist"] = np.std(data_list[identifier]["hist"]) # TODO: fix std calculation
                combined_data[identifier]["bin_edges"] = data_list[identifier]["bin_edges"][0]
        utils.save_object(combined_data, self.folder + "/combined.obj")


def plot_hist(link_data: str, axis: Axes | bool=True, identifiers = [], colors = [], std=False, mean=False, plot_kwargs = {}):
    fig, ax, data, identifiers, colors = utils.plot_setup(link_data, axis, identifiers, colors)

    if data["input_params"]["dimension"] != "Histogram":
        
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
        mean_value = data[identifier]["mean_angle"] if mean else None
        utils.plot_hist(ax, identifier, bin_edges, hist, colors[i % len(colors)], {}, std_hist, mean_value)

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Counts / frame") # TODO check normalization