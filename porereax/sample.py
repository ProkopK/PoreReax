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
from porereax.charge import ChargeSampler
from porereax.density import DensitySampler
from porereax.angle import AngleSampler
from porereax.bond_length import BondLengthSampler
from porereax.bond_structure import BondStructureSampler
from porereax.meta_sampler import Sampler, AtomSampler, BondSampler


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
        if "ovito" in sys.modules:
            print("Please remove ovito from loaded modules before using Sample class.")
            sys.exit(1)
        with mp.Pool() as pool:
            num_particles, num_frames, box = pool.apply_async(self.get_trajectory_data, (trajectory_file, bond_file, )).get()

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
        self.trajectory_file = trajectory_file
        self.bond_file = bond_file
        self.system = system

        self.sampler_inputs = {"charge_samplers": [],
                               "density_samplers": [],
                               "angle_samplers": [],
                               "bond_length_samplers": [],
                               "bond_structure_samplers": [],}
        self.samplers = []
        self.molecules = {}
        self.bonds = {}

        # Check atom library
        if not isinstance(atom_lib, dict):
            raise TypeError("atom_lib must be a dictionary mapping atom names to types.")
        if len(atom_lib) != len(set(atom_lib.values())) and len(atom_lib) != len(set(atom_lib.keys())):
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

        # TODO system

    @staticmethod
    def get_trajectory_data(trajectory_file, bond_file):
        """
        Extract trajectory metadata using Ovito.

        Parameters
        ----------
        trajectory_file : str
            Path to the trajectory file.
        bond_file : str, optional
            Path to the bond file.

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
        from ovito.data import BondsEnumerator
        os.environ["OVITO_THREAD_COUNT"] = "1"

        # Load trajectory
        if not isinstance(trajectory_file, str):
            raise TypeError("trajectory_file must be a string path to the trajectory file.")
        pipeline = import_file(trajectory_file)
        if bond_file:
            if not isinstance(bond_file, str):
                raise TypeError("bond_file must be a string path to the bond file.")
            bond_modifier = LoadTrajectoryModifier()
            bond_modifier.source.load(bond_file)
            pipeline.modifiers.append(bond_modifier)

        # Get and validate trajectory meta data
        first_frame = pipeline.compute()
        if first_frame.particles.count == 0:
            raise ValueError("No particles found in the trajectory file.")
        if first_frame.particles.bonds is None:
            raise ValueError("No bonds found. Ensure bond_file is provided or the trajectory contains bond data.")

        num_particles = first_frame.particles.count
        num_frames = pipeline.source.num_frames
        box = np.diagonal(first_frame.cell.matrix)

        return num_particles, num_frames, box

    def __add_sampler(self, sampler: Sampler):
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

    def add_charge_sampling(self, name_out, dimension, atoms, num_bins=800, range=(-2.0, 2.0)):
        """
        Add a ChargeSampler to the Sample instance.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Sampling dimension. Supported: "Histogram".
        atoms : list
            List of atom identifiers to sample.
        num_bins : int, optional
            Number of bins for histogram sampling.
        range : tuple, optional
            Range (min, max) for histogram sampling.
        """
        inputs = {"name_out": name_out,
                  "dimension": dimension,
                  "atoms": atoms,
                  "num_bins": num_bins,
                  "range": range,}
        self.sampler_inputs["charge_samplers"].append(inputs)

    def add_density_sampling(self, name_out, dimension, atoms, num_bins=200, direction="z", conditions={}):
        """
        Add a DensitySampler to the Sample instance.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Sampling dimension. Supported: "Time", "Cartesian1D", "Cartesian2D".
        atoms : list
            List of atom identifiers to sample.
        num_bins : int, optional
            Number of bins for position sampling. Relevant for Cartesian1D.
        direction : str, optional
            Direction along which to sample. For Cartesian1D, use ("x", "y", or "z").
        conditions : dict, optional
            Dictionary of conditions to filter atoms during sampling.
            Supported conditions:
            - "Charge": tuple (min_charge, max_charge) to filter atoms by charge.
        """
        inputs = {"name_out": name_out,
                  "dimension": dimension,
                  "atoms": atoms,
                  "num_bins": num_bins,
                  "direction": direction,
                  "conditions": conditions,}
        self.sampler_inputs["density_samplers"].append(inputs)

    def add_angle_sampling(self, name_out, dimension, atoms, num_bins=180, angle="all"):
        """
        Add an AngleSampler to the Sample instance.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Sampling dimension. Supported: "Histogram".
        atoms : list
            List of atom identifiers to sample.
        num_bins : int, optional
            Number of bins for histogram sampling.
        angle : str, optional
            Angle of interested atoms. Supported: "all", "A-B-C" where A, B, C are atom identifiers.
        """
        inputs = {"name_out": name_out,
                  "dimension": dimension,
                  "atoms": atoms,
                  "num_bins": num_bins,
                  "angle": angle,}
        self.sampler_inputs["angle_samplers"].append(inputs)

    def add_bond_length_sampling(self, name_out, dimension, bonds, num_bins=200, range=(0.0, 3.0)):
        """
        Add a BondLengthSampler to the Sample instance.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Sampling dimension. Supported: "Histogram".
        bonds : list
            List of bonds to sample. Each bond is defined as a dictionary in the format:
            {"bond": "a-b", "bonds_A": [...], "bonds_B": [...]} where a and b are atom identifiers,
            and bonds_A and bonds_B are lists of atom identifiers that atoms a and b are bonded to, respectively.
        num_bins : int, optional
            Number of bins for histogram sampling.
        range : tuple, optional
            Range (min, max) in Angstroms for histogram sampling.
        """
        inputs = {"name_out": name_out,
                  "dimension": dimension,
                  "bonds": bonds,
                  "num_bins": num_bins,
                  "range": range,}
        self.sampler_inputs["bond_length_samplers"].append(inputs)

    def add_bond_structure_sampling(self, name_out, dimension):
        """
        Add a BondStructureSampler to the Sample instance.

        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Sampling dimension. Supported: "BondStructure".
        """
        inputs = {"name_out": name_out,
                  "dimension": dimension,}
        self.sampler_inputs["bond_structure_samplers"].append(inputs)

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
        def add_bond_sampler(sampler: BondSampler):
            mols = sampler.get_mols()
            self.molecules.update(mols)
            bonds = sampler.get_bonds()
            self.bonds.update(bonds)
            self.__add_sampler(sampler)
        def add_atom_sampler(sampler: AtomSampler):
            mols = sampler.get_mols()
            self.molecules.update(mols)
            self.__add_sampler(sampler)
        for sampler_type in sampler_inputs:
            for sampler in sampler_inputs[sampler_type]:
                if sampler_type == "charge_samplers":
                    sampler_instance = ChargeSampler(name_out=sampler["name_out"],
                                                     dimension=sampler["dimension"],
                                                     atoms=sampler["atoms"],
                                                     process_id=process_id,
                                                     atom_lib=self.name_to_type, 
                                                     masses=self.masses,
                                                     num_frames=self.num_frames,
                                                     box=self.box,
                                                     num_bins=sampler["num_bins"],
                                                     range=sampler["range"])
                    add_atom_sampler(sampler_instance)
                elif sampler_type == "density_samplers":
                    sampler_instance = DensitySampler(name_out=sampler["name_out"],
                                                      dimension=sampler["dimension"],
                                                      atoms=sampler["atoms"],
                                                      process_id=process_id,
                                                      atom_lib=self.name_to_type, 
                                                      masses=self.masses,
                                                      num_frames=self.num_frames,
                                                      box=self.box,
                                                      num_bins=sampler["num_bins"],
                                                      direction=sampler["direction"],
                                                      conditions=sampler["conditions"])
                    add_atom_sampler(sampler_instance)
                elif sampler_type == "angle_samplers":
                    sampler_instance = AngleSampler(name_out=sampler["name_out"],
                                                    dimension=sampler["dimension"],
                                                    atoms=sampler["atoms"],
                                                    process_id=process_id,
                                                    atom_lib=self.name_to_type, 
                                                    masses=self.masses,
                                                    num_frames=self.num_frames,
                                                    box=self.box,
                                                    num_bins=sampler["num_bins"],
                                                    angle=sampler["angle"])
                    add_atom_sampler(sampler_instance)
                elif sampler_type == "bond_length_samplers":
                    sampler_instance = BondLengthSampler(name_out=sampler["name_out"],
                                                         dimension=sampler["dimension"],
                                                         bonds=sampler["bonds"],
                                                         process_id=process_id,
                                                         atom_lib=self.name_to_type, 
                                                         masses=self.masses,
                                                         num_frames=self.num_frames,
                                                         box=self.box,
                                                         num_bins=sampler["num_bins"],
                                                         range=sampler["range"])
                    add_bond_sampler(sampler_instance)
                elif sampler_type == "bond_structure_samplers":
                    sampler_instance = BondStructureSampler(name_out=sampler["name_out"],
                                                            dimension=sampler["dimension"],
                                                            process_id=process_id,
                                                            atom_lib=self.name_to_type, 
                                                            masses=self.masses,
                                                            num_frames=self.num_frames,
                                                            box=self.box)
                    add_atom_sampler(sampler_instance)

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

        self.init_samplers(self.sampler_inputs, process_id=-1)

        if is_parallel and num_cores > 1:
            frames_per_core = np.array_split(self.frames, num_cores)
            start_end_nthframe_list = [(frames[0], frames[-1], 1) for frames in frames_per_core]
            for i, (start_frame, end_frame, _) in enumerate(start_end_nthframe_list):
                print(f"Process {i}: frames {start_frame} to {end_frame}")
            if "ovito" in sys.modules:
                print("Ovito module detected. Please remove it before using parallel sampling. This exit is intentional to avoid infinite spawning of subprocesses.")
                sys.exit(1)
            print(f"Starting parallel sampling with {num_cores} cores...")
            with mp.Pool(num_cores) as pool:
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
            print([r.get() for r in results])
        else:
            print("Starting serial sampling...")
            self.sample_helper()

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
        molecules_per_atom_type = {}
        for atom_type in self.type_to_name:
            molecules_per_atom_type[atom_type] = []
            for identifier in self.molecules:
                if self.molecules[identifier]["atom"] == atom_type:
                    bonds = self.molecules[identifier]["bonds"]
                    molecules_per_atom_type[atom_type].append((bonds, identifier))
            # Sort by number of bonds (fewest first)
            molecules_per_atom_type[atom_type].sort(key=lambda x: len(x[1]))
            if molecules_per_atom_type[atom_type] == []:
                molecules_per_atom_type.pop(atom_type)
        # List for each molecule, which atoms belong to it
        molecule_idx = {}
        # List for each molecule, the bonded atoms of each atom
        molecule_bonds = {}
        for identifier in self.molecules:
            molecule_idx[identifier] = np.zeros(self.num_particles, dtype=bool)
            if self.molecules[identifier]["bonds"] != None:
                molecule_bonds[identifier] = np.zeros((self.num_particles, len(self.molecules[identifier]["bonds"][0]), ), dtype=int)
            else:
                molecule_bonds[identifier] = np.zeros((self.num_particles, 0, ), )

        bond_idx = {}

        # Loop over frames
        for frame_idx in self.frames:
            print(f"Processing frame {frame_idx}...")
            frame = self.pipeline.compute(frame_idx)
            atom_types = frame.particles.particle_types.array
            atom_charges = frame.particles.get("Charge").array if "Charge" in frame.particles else np.zeros(self.num_particles)
            atom_identifiers = frame.particles.identifiers.array if frame.particles.identifiers is not None else np.arange(self.num_particles)
            atom_positions = frame.particles.positions.array
            # atom_velocities = frame.particles.velocities.array
            bond_count = frame.particles.bonds.count
            bond_topology = frame.particles.bonds.topology.array
            bond_enum = BondsEnumerator(frame.particles.bonds)

            # Reset molecule indices
            for mol in molecule_idx:
                molecule_idx[mol] = np.zeros(self.num_particles, dtype=bool)
                molecule_bonds[mol] = np.zeros((self.num_particles, molecule_bonds[mol].shape[1], ), dtype=int)

            # Identify molecules
            for atom_type in molecules_per_atom_type:
                atoms = np.where(atom_types == atom_type)[0]
                # Molecule registered without bond constraints; it should be first because of sorting
                if molecules_per_atom_type[atom_type][0][0] == None:
                    molecule_idx[molecules_per_atom_type[atom_type][0][1]][atoms] = 1
                    # No other molecules of this atom type
                    if len(molecules_per_atom_type[atom_type]) == 1:
                        continue
                # Atom with bond constraints
                for atom in atoms:
                    bonds = list(bond_enum.bonds_of_particle(atom))
                    particles = bond_topology[bonds].flatten()
                    other_particles = particles[particles != atom]
                    other_types = list(atom_types[other_particles])
                    for bond_permutations, identifier in molecules_per_atom_type[atom_type]:
                        if bond_permutations != None and other_types in bond_permutations:
                            molecule_idx[identifier][atom] = 1
                            molecule_bonds[identifier][atom] = other_particles
            for mol in molecule_idx:
                molecule_idx[mol] = np.where(molecule_idx[mol])[0]
                molecule_bonds[mol] = molecule_bonds[mol][molecule_idx[mol]]

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
                        if atom_a in molecule_idx[mol_A] and atom_b in molecule_idx[mol_B]:
                            bond_idx[identifier][bond_id] = 1
                    elif ((type_a == bond_def[1] and type_b == bond_def[0])):
                        if atom_a in molecule_idx[mol_B] and atom_b in molecule_idx[mol_A]:
                            bond_idx[identifier][bond_id] = 1
            for identifier in bond_idx:
                bond_idx[identifier] = np.where(bond_idx[identifier])[0]

            # Sampling
            for sampler in self.samplers:
                sampler.sample(frame=frame_idx-self.start_frame,
                               positions=atom_positions,
                               charges=atom_charges,
                               mol_index=molecule_idx,
                               mol_bonds=molecule_bonds,
                               atom_types=atom_types,
                               bond_index=bond_idx,
                               bond_enum=bond_enum,
                               bond_topology=bond_topology,
                )

        for sampler in self.samplers:
            input_params, data = sampler.get_data()
            data.update({"input_params": input_params})
            file_name = sampler.file_out
            utils.save_object(data, file_name)