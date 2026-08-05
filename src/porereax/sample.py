"""
Module providing the Sample class for molecular trajectory sampling.

This module defines the Sample class, which manages the sampling of molecular
trajectories using various samplers. It supports parallel processing using
the multiprocessing module and integrates with the Ovito library for trajectory
handling. The Sample class allows users to add different samplers, configure
sampling parameters, and execute the sampling process either in parallel or
serially.
"""

import numpy as np
import multiprocessing as mp
import os
import sys
import porereax.utils as utils

from collections.abc import Callable
from numpy.typing import NDArray
from porereax.charge import ChargeSampler
from porereax.density import DensitySampler, BondDensitySampler, ReactionSampler
from porereax.angle import AngleSampler
from porereax.bond_length import BondLengthSampler
from porereax.molecule_structure import MoleculeStructureSampler
from porereax.rdf import RdfSampler
from porereax.meta_sampler import Sampler


type Region = str | Callable[[NDArray[np.float64]], NDArray[np.bool_]]


class Sample:
    """
    Class to manage sampling of molecular trajectories.
    """

    def __init__(self, atom_lib, masses, trajectory_file, bond_file=None, system=None, start_end_nthframe=(0, -1, 1)):
        """
        Initialize Sample instance.

        To create a Sample instance, ensure that the 'ovito' module is not
        already imported in the current Python session, as it may lead to
        conflicts during parallel processing.

        Parameters
        ----------
        atom_lib : dict
            Library mapping atom names to types.
        masses : dict
            Dictionary mapping atom names to their masses.
        trajectory_file : str
            Path to the trajectory file.
        bond_file : str, optional
            Path to the bond file.
        system : object, optional
            System object containing additional information.
        start_end_nthframe : tuple, optional
            Tuple specifying (start_frame, end_frame, nth_frame) for sampling.
        """
        if "forkserver" in mp.get_all_start_methods():
            ctx = mp.get_context("forkserver")
        elif "spawn" in mp.get_all_start_methods():
            ctx = mp.get_context("spawn")
        else:
            ctx = mp.get_context("fork")
            if "ovito" in sys.modules:
                raise RuntimeError(
                    "The 'ovito' module is already imported. Please remove it from the "
                    "loaded modules to avoid conflicts during parallel processing. "
                    "This is necessary because the OS does not support the 'spawn' start method."
                )
        with ctx.Pool(1) as pool:
            num_particles, num_frames, box = pool.apply_async(self.get_trajectory_data, (trajectory_file, bond_file, atom_lib, )).get()

        print(f"Trajectory has {num_particles} particles and {num_frames} frames.")

        start_frame, end_frame, nth_frame = start_end_nthframe

        self.init_helper(atom_lib, masses, trajectory_file, bond_file, system, start_frame, end_frame, nth_frame, num_particles, num_frames, box)

    def init_helper(self, atom_lib, masses, trajectory_file, bond_file, system, start_frame, end_frame, nth_frame, num_particles, num_frames, box):
        """
        Helper function to initialize Sample instance.

        Parameters
        ----------
        atom_lib : dict
            Library mapping atom names to types.
        masses : dict
            Dictionary mapping atom names to their masses.
        trajectory_file : str
            Path to the trajectory file.
        bond_file : str, optional
            Path to the bond file.
        system : object, optional
            System object containing additional information.
        start_frame : int
            Starting frame for sampling.
        end_frame : int
            Ending frame for sampling.
        nth_frame : int
            Step size for frame sampling.
        num_particles : int
            Number of particles in the trajectory.
        num_frames : int
            Total number of frames in the trajectory.
        box : np.ndarray
            Simulation box dimensions.
        """
        self.trajectory_file = os.path.abspath(trajectory_file)
        self.bond_file = os.path.abspath(bond_file) if bond_file else None
        self.system = system # is only used to pass to the subprocesses

        self.sampler_inputs = {"charge_samplers": [],
                               "density_samplers": [],
                               "bond_density_samplers": [],
                               "angle_samplers": [],
                               "bond_length_samplers": [],
                               "molecule_structure_samplers": [],
                               "rdf_samplers": [],
                               "reaction_samplers": [],
                               }
        # Registry mapping sampler_inputs key -> (SamplerClass, kind, extra_kwarg_keys)
        self._SAMPLER_REGISTRY = {
            "charge_samplers":            (ChargeSampler,           "atom", ["atoms", "num_bins", "range"]),
            "density_samplers":           (DensitySampler,          "atom", ["atoms", "num_bins", "direction", "conditions"]),
            "bond_density_samplers":      (BondDensitySampler,      "bond", ["bonds", "num_bins", "direction", "conditions"]),
            "angle_samplers":             (AngleSampler,            "atom", ["atoms", "num_bins", "angle"]),
            "bond_length_samplers":       (BondLengthSampler,       "bond", ["bonds", "num_bins", "range"]),
            "molecule_structure_samplers":(MoleculeStructureSampler,"atom", []),
            "rdf_samplers":               (RdfSampler,              "atom", ["pairs", "num_bins", "r_max"]),
            "reaction_samplers":          (ReactionSampler,         "atom", ["reactions", "num_bins", "direction", "position"]),
        }

        self.samplers = []
        self.molecules = {}
        self.bonds = {}

        # Check atom library
        if not isinstance(atom_lib, dict):
            raise TypeError("atom_lib must be a dictionary mapping atom names to types.")
        if len(atom_lib) != len(set(atom_lib.values())) or len(atom_lib) != len(set(atom_lib.keys())):
            raise ValueError("atom_lib must have a one-to-one mapping of atom names to types.")
        self.type_to_name = {v: k for k, v in atom_lib.items()}
        self.name_to_type = atom_lib

        # Check and set masses
        if not isinstance(masses, dict):
            raise TypeError("masses must be a dictionary mapping atom names to masses.")
        if set(masses.keys()) != set(atom_lib.keys()):
            raise ValueError("masses keys must match atom_lib keys.")
        self.masses = masses

        # Validate start and end frame values
        if (start_frame < 0) or (end_frame < -1) or (start_frame > end_frame and end_frame != -1) or (start_frame >= num_frames) or (end_frame >= num_frames):
            raise ValueError(f"Invalid start_end frame range. The trajectory has {num_frames} frames and the provided range is ({start_frame}, {end_frame}).")

        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame != -1 else num_frames - 1
        self.nth_frame = nth_frame
        self.num_particles = num_particles
        self.box = box
        self.frames = range(self.start_frame, self.end_frame + 1, self.nth_frame)
        self.num_frames = len(self.frames)

        self.system_properties = utils.read_pore_yml(system) if system else None

    @staticmethod
    def get_trajectory_data(trajectory_file, bond_file, atom_lib):
        """
        Extract trajectory metadata using Ovito.

        Parameters
        ----------
        trajectory_file : str
            Path to the trajectory file.
        bond_file : str, optional
            Path to the bond file.
        atom_lib : dict
            Library mapping atom names to types.

        Returns
        -------
        num_particles : int
            Number of particles in the trajectory.
        num_frames : int
            Total number of frames in the trajectory.
        box : np.ndarray
            Simulation box dimensions.
        """
        from ovito.io import import_file
        from ovito.modifiers import LoadTrajectoryModifier
        os.environ["OVITO_THREAD_COUNT"] = "1"

        # Load trajectory
        if not os.path.isfile(trajectory_file):
            raise FileNotFoundError(f"Trajectory file '{trajectory_file}' not found.")
        pipeline = import_file(trajectory_file)
        if bond_file:
            if not os.path.isfile(bond_file):
                raise FileNotFoundError(f"Bond file '{bond_file}' not found.")
            bond_modifier = LoadTrajectoryModifier()
            bond_modifier.source.load(bond_file)
            pipeline.modifiers.append(bond_modifier)

        # Get and validate trajectory meta data
        first_frame = pipeline.compute()
        if first_frame.particles.count == 0:
            raise ValueError("No particles found in the trajectory file.")
        if first_frame.particles.bonds is None:
            raise ValueError("No bonds found. Ensure bond_file is provided or the trajectory contains bond data.")
        type_set = set(first_frame.particles.particle_types.array)
        atom_type_set = set(atom_lib.values())
        if type_set != atom_type_set:
            raise ValueError(f"Atom types in trajectory {type_set} do not match those in atom_lib {atom_type_set}.")

        num_particles = first_frame.particles.count
        num_frames = pipeline.source.num_frames
        box = np.diagonal(first_frame.cell.matrix)

        return num_particles, num_frames, box

    # --------------------------- Add Sampler Methods ---------------------------
    def add_molecule_structure_sampling(self, name_out: str, region: Region = "Box"):
        """
        Add sampling for molecule structures to analyse the bonding of atoms and identify substructures.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        """
        dimension = "MoleculeStructure"
        inputs = {"name_out": name_out,
                  "dimension": dimension,
                  "region": region,}
        self.sampler_inputs["molecule_structure_samplers"].append(inputs)

    def add_charge_sampling(self, name_out: str, atoms: list[dict], region: Region = "Box", num_bins=800, range=(-2.0, 2.0)):
        """
        Add sampling for charge distribution of the central atom in the specified atom structures.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        atoms : list
            List of atom structures to sample. Each atom structure is defined as a dictionary in the format:
            {"atom": "a", "bonds": [b, b, c, ...]}, where a is the central atom and b, c, ... are the bonded atoms. With a, b, c being atom identifiers. The order of atoms in the "bonds" list does not matter. The "bonds" list can be empty to indicate that the atom is not bonded to any other atoms. If the dictionary does not contain the "bonds" key, every atom of type a will be sampled regardless of its bonding environment.
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        num_bins : int, optional
            Number of bins for the histogram. Default is 800.
        range : tuple, optional
            Range (min, max) in e for which to compute the histogram. Default is (-2.0, 2.0).
        """
        dimension = "Histogram"
        inputs = {"name_out": name_out,
                  "atoms": atoms,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "range": range,}
        self.sampler_inputs["charge_samplers"].append(inputs)

    def add_density_sampling(self, name_out: str, atoms: list[dict], dimension: str, region: Region = "Box", num_bins=200, direction="z", conditions={}):
        """
        Add sampling for time or position density distribution of the specified atom structures. For positional sampling the position of the central atom is used.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        atoms : list
            List of atom structures to sample. Each atom structure is defined as a dictionary in the format:
            {"atom": "a", "bonds": [b, b, c, ...]}, where a is the central atom and b, c, ... are the bonded atoms. With a, b, c being atom identifiers. The order of atoms in the "bonds" list does not matter. The "bonds" list can be empty to indicate that the atom is not bonded to any other atoms. If the dictionary does not contain the "bonds" key, every atom of type a will be sampled regardless of its bonding environment.
        dimension : str
            Sampling dimension. Supported: "Time", "Cartesian1D", "Cartesian2D".
            - "Time": Samples the amount of atom structures over time.
            - "Cartesian1D": Samples the amount of atom structures along a specified direction (x, y, or z) in the simulation box.
            - "Cartesian2D": Samples the amount of atom structures in a 2D plane (xy, xz, or yz) in the simulation box.
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        num_bins : int, optional
            Number of bins for position sampling. Not used for time sampling.
        direction : str, optional
            Direction along which to sample. Options depending on the dimension:
            - For "Cartesian1D": use ("x", "y", or "z").
            - For "Cartesian2D": use ("xy", "xz", or "yz").
        conditions : dict, optional
            Dictionary of conditions to filter atoms during sampling.
            Supported conditions:
            - "Charge": tuple (min_charge, max_charge) to filter atoms by charge.
            - "Angle": tuple (min_angle, max_angle) to filter atoms by angle formed with bonded atoms.
        """
        inputs = {"name_out": name_out,
                  "atoms": atoms,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "direction": direction,
                  "conditions": conditions,}
        self.sampler_inputs["density_samplers"].append(inputs)

    def add_angle_sampling(self, name_out: str, atoms: list[dict], region: Region = "Box", num_bins=180, angle="all"):
        """
        Add sampling for angle distribution of the specified atom structures. The angle is defined by the central atom and its bonded atoms. 

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        atoms : list
            List of atom structures to sample. Each atom structure is defined as a dictionary in the format:
            {"atom": "a", "bonds": [b, b, c, ...]}, where a is the central atom and b, c, ... are the bonded atoms. With a, b, c being atom identifiers. The order of atoms in the "bonds" list does not matter. The "bonds" list can be empty to indicate that the atom is not bonded to any other atoms. If the dictionary does not contain the "bonds" key, every atom of type a will be sampled regardless of its bonding environment.
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        num_bins : int, optional
            Number of bins for the histogram. Default is 180.
        angle : str, optional
            Angle of interested atoms. Supported: "all", "a-b-c"
            - "all": Samples all angles formed by the central atom and all of its bonded atoms.
            - "a-b-c": Samples only the angle formed by the central atom (b) and two specific bonded atoms (a and c), regardless of other bonded atoms d, ... With a, b, c being atom identifiers. 
        """
        dimension = "Histogram"
        inputs = {"name_out": name_out,
                  "atoms": atoms,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "angle": angle,}
        self.sampler_inputs["angle_samplers"].append(inputs)

    def add_bond_density_sampling(self, name_out: str, bonds: list[dict], dimension: str, region: Region = "Box", num_bins=200, direction="z", conditions={}):
        """
        Add sampling for time or position density distribution of the specified bonds. For positional sampling the position of the bond center is used.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        bonds : list
            List of bonds to sample. Each bond is defined as a dictionary in the format:
            {"bond": "a-b", "bonds_A": [c, ...], "bonds_B": [d, ...]}, where a and b are the bonded atoms, and c, d, ... are the atoms bonded to A and B, respectively. With a, b, c, d being atom identifiers. Atom b/a does not need to be added to the "bonds_A"/"bonds_B" list. The order of atoms in the "bonds_A" and "bonds_B" lists does not matter. The "bonds_A" and "bonds_B" lists can be empty to indicate that the atoms a and b are not bonded to any other atoms. If the dictionary does not contain the "bonds_A" or "bonds_B" keys, every bond of type a-b will be sampled regardless of its bonding environment.
        dimension : str
            Sampling dimension. Supported: "Time", "Cartesian1D", "Cartesian2D".
            - "Time": Samples the amount of the specified bonds over time.
            - "Cartesian1D": Samples the amount of the specified bonds along a specified direction (x, y, or z) in the simulation box.
            - "Cartesian2D": Samples the amount of the specified bonds in a 2D plane (xy, xz, or yz) in the simulation box.
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        num_bins : int, optional
            Number of bins for position sampling. Not used for time sampling.
        direction : str, optional
            Direction along which to sample. Options depending on the dimension:
            - For "Cartesian1D": use ("x", "y", or "z").
            - For "Cartesian2D": use ("xy", "xz", or "yz").
        conditions : dict, optional
            Dictionary of conditions to filter bonds during sampling.
            Supported conditions:
            - "Bond Length": tuple (min_len, max_len) to filter bonds by bond length.
        """
        inputs = {"name_out": name_out,
                  "bonds": bonds,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "direction": direction,
                  "conditions": conditions,}
        self.sampler_inputs["bond_density_samplers"].append(inputs)

    def add_bond_length_sampling(self, name_out: str, bonds: list[dict], dimension: str, region: Region = "Box", num_bins=200, range=(0.0, 3.0)):
        """
        Add sampling for bond length or bond order distribution of the specified bonds.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        bonds : list
            List of bonds to sample. Each bond is defined as a dictionary in the format:
            {"bond": "a-b", "bonds_A": [c, ...], "bonds_B": [d, ...]}, where a and b are the bonded atoms, and c, d, ... are the atoms bonded to A and B, respectively. With a, b, c, d being atom identifiers. Atom b/a does not need to be added to the "bonds_A"/"bonds_B" list. The order of atoms in the "bonds_A" and "bonds_B" lists does not matter. The "bonds_A" and "bonds_B" lists can be empty to indicate that the atoms a and b are not bonded to any other atoms. If the dictionary does not contain the "bonds_A" or "bonds_B" keys, every bond of type a-b will be sampled regardless of its bonding environment.
        dimension : str
            Sampling dimension. Supported: "Bond Length" and "Bond Order"
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        num_bins : int, optional
            Number of bins for the histogram. Default is 200.
        range : tuple, optional
            Range (min, max) for which to compute the histogram. Default is (0.0, 3.0).
            - For "Bond Length": range is in Angstroms.
            - For "Bond Order": range is in bond order units defined by the ReaxFF force field.
        """
        inputs = {"name_out": name_out,
                  "bonds": bonds,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "range": range,}
        self.sampler_inputs["bond_length_samplers"].append(inputs)

    def add_rdf_sampling(self, name_out: str, pairs: list[tuple[dict, dict]], region: Region = "Box", num_bins=200, r_max=7.0):
        """
        Add sampling for radial distribution function (RDF) of the specified atom pairs.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        pairs : list
            List of atom pairs to sample. Each pair is defined as a tuple of two dictionaries in the format:
            ({"atom": "a", "bonds": [...]}, {"atom": "b", "bonds": [...]}) where a and b are atom identifiers,
            and bonds are lists of atom identifiers that atoms a and b are bonded to, respectively. Each dictionary works the same way as other samplers, with `atoms` as a parameter.
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
        num_bins : int, optional
            Number of bins for the histogram. Default is 200.
        r_max : float, optional
            Maximum distance in Angstroms for which to compute the histogram. Default is 7.0.
            Be aware that the maximum distance significantly affects the computation time.
        """
        dimension = "Histogram"
        inputs = {"name_out": name_out,
                  "pairs": pairs,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "r_max": r_max,}
        self.sampler_inputs["rdf_samplers"].append(inputs)

    def add_reaction_sampling(self, name_out: str, reactions: list[tuple], dimension: str, region: Region = "Box", num_bins=200, direction="z", position="center"):
        """
        Add sampling for reaction events of the specified reactions.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        reactions : list
            List of reactions to sample. Each reaction is defined as a tuple of two dictionaries in the format:
            ({"atom": "a", "bonds": [...]}, {"atom": "b", "bonds": [...]}) where a and b are atom identifiers,
            and bonds are lists of atom identifiers that atoms a and b are bonded to, respectively. Each dictionary works the same way as other samplers, with `atoms` as a parameter.
        dimension : str
            Sampling dimension. Supported: "Time", "Cartesian1D", "Cartesian2D".
            - "Time": Samples the amount of the specified reactions over time.
            - "Cartesian1D": Samples the amount of the specified reactions along a specified direction (x, y, or z) in the simulation box.
            - "Cartesian2D": Samples the amount of the specified reactions in a 2D plane (xy, xz, or yz) in the simulation box.
        region : Region, optional
            Region of the box to sample. Supported: "Box" or a user-defined 
            function that takes atom positions (N, 3) as input and returns a boolean mask (N,).
        num_bins : int, optional
            Number of bins for position sampling. Default is 200.
        direction : str, optional
            Direction along which to sample. Options depending on the dimension:
            - For "Cartesian1D": use ("x", "y", or "z").
            - For "Cartesian2D": use ("xy", "xz", or "yz").
        position : str, optional
            Position of the reaction event to sample. Supported: "center", "reactant", "product"
            - "center": Samples the position of the reaction event at the center between the reactant and product atoms.
            - "reactant": Samples the position of the reaction event at the position of the reactant atoms.
            - "product": Samples the position of the reaction event at the position of the product atoms.
        """
        inputs = {"name_out": name_out,
                  "reactions": reactions,
                  "dimension": dimension,
                  "region": region,
                  "num_bins": num_bins,
                  "direction": direction,
                  "position": position,}
        self.sampler_inputs["reaction_samplers"].append(inputs)

    def _add_sampler(self, sampler: Sampler):
        """
        Add a sampler to the Sample instance.

        Parameters
        ----------
        sampler : Sampler
            An instance of a Sampler subclass to be added
        """
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be an instance of Sampler class.")
        self.samplers.append(sampler)

    def init_samplers(self, sampler_inputs, process_id):
        """
        Initialize samplers based on provided configurations.

        Parameters
        ----------
        sampler_inputs : dict
            Dictionary of sampler input configurations.
        process_id : int
            Process ID for parallel sampling.
        """
        common_kwargs = dict(
            process_id=process_id,
            atom_lib=self.name_to_type,
            masses=self.masses,
            num_frames=self.num_frames,
            box=self.box,
            system_properties=self.system_properties,
        )

        for sampler_type, sampler_configs in sampler_inputs.items():
            if sampler_type not in self._SAMPLER_REGISTRY:
                raise ValueError(f"Unknown sampler type: '{sampler_type}'")
            SamplerClass, kind, extra_keys = self._SAMPLER_REGISTRY[sampler_type]
            for cfg in sampler_configs:
                kwargs = dict(
                    name_out=cfg["name_out"],
                    dimension=cfg["dimension"],
                    region=cfg["region"],
                    **common_kwargs,
                    **{k: cfg[k] for k in extra_keys},
                )
                instance = SamplerClass(**kwargs)

                # Add sampler instance to the list of samplers
                if kind == "bond":
                    self.bonds.update(instance.get_bonds())
                self.molecules.update(instance.get_mols())
                self._add_sampler(instance)

    def sample(self, is_parallel=True, num_cores=0):
        """
        Execute the sampling process.

        Parameters
        ----------
        is_parallel : bool, optional
            Whether to use parallel processing.
        num_cores : int, optional
            Number of CPU cores to use for parallel processing.
        """
        # Determine number of cores to use
        avail_cores = mp.cpu_count()
        cluster_tasks = (
            os.getenv("SLURM_NTASKS")
            or os.getenv("PBS_NP")
            or os.getenv("LSB_DJOB_NUMPROC")
            or os.getenv("NSLOTS"))
        cluster_tasks = int(cluster_tasks) if cluster_tasks else None
        max_cores = min(avail_cores, cluster_tasks, self.num_frames) if cluster_tasks else min(avail_cores-1, self.num_frames)
        num_cores = num_cores if num_cores and num_cores<=max_cores else max_cores

        # Run initialization of samplers in the main process to catch any potential issues before spawning subprocesses
        self.init_samplers(self.sampler_inputs, process_id=-1)

        if is_parallel and num_cores > 1:
            frames_per_core = np.array_split(self.frames, num_cores)
            start_end_nthframe_list = [(frames[0], frames[-1], 1) for frames in frames_per_core]
            for i, (start_frame, end_frame, _) in enumerate(start_end_nthframe_list):
                print(f"Process {i}: frames {start_frame} to {end_frame}")
            print(f"Starting parallel sampling with {num_cores} cores...")
            if "forkserver" in mp.get_all_start_methods():
                ctx = mp.get_context("forkserver")
            elif "spawn" in mp.get_all_start_methods():
                ctx = mp.get_context("spawn")
            else:
                ctx = mp.get_context("fork")
                if "ovito" in sys.modules:
                    raise RuntimeError(
                        "The 'ovito' module is already imported. Please remove it from the "
                        "loaded modules to avoid conflicts during parallel processing. "
                        "This is necessary because the OS does not support the 'spawn' start method."
                    )
            with ctx.Pool(num_cores) as pool:
                results = [pool.apply_async(self.init_subprocess_sampler, (self.name_to_type,
                                                                           self.masses,
                                                                           self.trajectory_file,
                                                                           self.bond_file,
                                                                           self.system,
                                                                           start_end_nthframe_list[process_id],
                                                                           self.sampler_inputs,
                                                                           process_id,
                                                                           self.num_particles,
                                                                           np.inf,
                                                                           self.box
                                                                           )) for process_id in range(num_cores)]
                pool.close()
                pool.join()
            # print([r.get() for r in results])
            print("Parallel sampling completed.")
        else:
            print("Starting serial sampling...")
            self.sample_helper()
            print("Serial sampling completed.")

        for sampler in self.samplers:
            sampler.join_samplers(num_cores=num_cores if is_parallel else 1)

    @staticmethod
    def init_subprocess_sampler(atom_lib, masses, trajectory_file, bond_file, system, start_end_nthframe, sampler_inputs, process_id, num_particles, num_frames, box):
        """
        Initialize and run sampling in a subprocess.

        This static method is designed to be called within a subprocess for parallel sampling.

        Parameters
        ----------
        atom_lib : dict
            Library mapping atom names to types.
        masses : dict
            Dictionary mapping atom names to their masses.
        trajectory_file : str
            Path to the trajectory file.
        bond_file : str, optional
            Path to the bond file.
        system : object, optional
            System object containing additional information.
        start_end_nthframe : tuple
            Tuple specifying (start_frame, end_frame, nth_frame) for sampling.
        sampler_inputs : dict
            Dictionary of sampler input configurations.
        process_id : int
            Process ID for parallel sampling.
        num_particles : int
            Number of particles in the trajectory.
        box : np.ndarray
            Simulation box dimensions.

        Returns
        -------
        str
            Completion message for the subprocess.
        """
        sample_instance = Sample.__new__(Sample)
        start_frame, end_frame, nth_frame = start_end_nthframe
        sample_instance.init_helper(atom_lib, masses, trajectory_file, bond_file, system, start_frame, end_frame, nth_frame, num_particles, num_frames, box)
        sample_instance.init_samplers(sampler_inputs, process_id)
        sample_instance.sample_helper()
        return f"Process {process_id} finished sampling."

    def sample_helper(self):
        """
        Helper function to perform the sampling process.
        """
        from ovito.io import import_file
        from ovito.modifiers import LoadTrajectoryModifier
        from ovito.data import BondsEnumerator
        os.environ["OVITO_THREAD_COUNT"] = "1"


        # Load trajectory
        self.pipeline = import_file(self.trajectory_file)
        if self.bond_file:
            bond_modifier = LoadTrajectoryModifier()
            bond_modifier.source.load(self.bond_file)
            self.pipeline.modifiers.append(bond_modifier)

        # Prepare molecule indexing
        # Example: for one atom type: [('O', None), ('O()', [[]]), ('O(H+H)', [[1, 1]]), ('O(H+Si)', [[1, 2], [2, 1]])]
        molecules_per_atom_type = {}
        for atom_type in self.type_to_name:
            molecules_per_atom_type[atom_type] = []
            for identifier in self.molecules:
                if self.molecules[identifier]["atom"] == atom_type:
                    bonds = self.molecules[identifier]["bonds"]
                    molecules_per_atom_type[atom_type].append((identifier, bonds))
            # Sort molecules: first those without bond constraints, then by increasing number of bond constraints
            molecules_per_atom_type[atom_type].sort(key=lambda x: len(x[1][0]) if x[1] is not None else -1)
            # Remove atom types without registered molecules
            if not molecules_per_atom_type[atom_type]:
                molecules_per_atom_type.pop(atom_type)
        molecule_idx = {} # Mask, that indicates for each molecule, which atoms belong to it. Shape: (num_particles, ) type: bool
        molecule_bonds = {} # Mapping of molecule to the id of atoms it is bonded to. Shape: (num_particles, num_bonds_per_molecule) type: int
        for identifier in self.molecules:
            molecule_idx[identifier] = np.zeros(self.num_particles, dtype=bool)
            if self.molecules[identifier]["bonds"] is not None:
                molecule_bonds[identifier] = np.zeros((self.num_particles, len(self.molecules[identifier]["bonds"][0]), ), dtype=int)
            else:
                molecule_bonds[identifier] = np.zeros((self.num_particles, 0, ), )

        bond_idx = {}

        # Loop over frames
        for frame_idx in self.frames:
            print(f"Processing frame {frame_idx}...")
            frame = self.pipeline.compute(frame_idx)
            atom_types = frame.particles.particle_types.array
            bond_count = frame.particles.bonds.count
            bond_topology = frame.particles.bonds.topology.array
            bond_enum = BondsEnumerator(frame.particles.bonds)

            # Molecule information
            # Reset molecule indices
            for mol in molecule_idx:
                molecule_idx[mol][:] = 0
                molecule_bonds[mol][:] = 0

            # Identify molecules
            for atom_type in molecules_per_atom_type:
                atoms = np.where(atom_types == atom_type)[0]
                # Molecule registered without bond constraints; it is first because of sorting
                if molecules_per_atom_type[atom_type][0][1] is None:
                    molecule_idx[molecules_per_atom_type[atom_type][0][0]][atoms] = 1
                    # No other molecules of this atom type
                    if len(molecules_per_atom_type[atom_type]) == 1:
                        continue
                # Atom with bond constraints
                for atom in atoms:
                    bonds = list(bond_enum.bonds_of_particle(atom))
                    particles = bond_topology[bonds].flatten()
                    other_particles = particles[particles != atom]
                    other_types = list(atom_types[other_particles])
                    for identifier, bond_permutations in molecules_per_atom_type[atom_type]:
                        if bond_permutations is not None and other_types in bond_permutations:
                            molecule_idx[identifier][atom] = 1
                            molecule_bonds[identifier][atom] = other_particles

            # Bond information
            # Reset bond indices
            for identifier in self.bonds:
                bond_idx[identifier] = np.zeros(bond_count, dtype=bool)

            # Identify bonds
            for bond_id, bond in enumerate(bond_topology):
                atom_a = bond[0]
                atom_b = bond[1]
                type_a = atom_types[atom_a]
                type_b = atom_types[atom_b]
                for identifier in self.bonds:
                    bond_info = self.bonds[identifier]
                    bond_def = bond_info["bond"]
                    mol_A = bond_info["mol_A"]
                    mol_B = bond_info["mol_B"]
                    if ((type_a == bond_def[0] and type_b == bond_def[1])):
                        if molecule_idx[mol_A][atom_a] and molecule_idx[mol_B][atom_b]:
                            bond_idx[identifier][bond_id] = 1
                    elif ((type_a == bond_def[1] and type_b == bond_def[0])):
                        if molecule_idx[mol_A][atom_b] and molecule_idx[mol_B][atom_a]:
                            bond_idx[identifier][bond_id] = 1
            for identifier in bond_idx:
                bond_idx[identifier] = np.where(bond_idx[identifier])[0]

            # Sampling
            for sampler in self.samplers:
                sampler.sample(frame_id=frame_idx-self.start_frame,
                               mol_index=molecule_idx,
                               mol_bonds=molecule_bonds,
                               bond_index=bond_idx,
                               frame=frame,
                               bond_enum=bond_enum,
                )

        for sampler in self.samplers:
            sampler.save_object()
