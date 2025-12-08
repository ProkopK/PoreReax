import numpy as np
import multiprocessing as mp
import os
import sys

import porereax.utils as utils
from porereax.charge import ChargeSampler
from porereax.utils import Sampler, AtomSampler, BondSampler


class Sample:
    def __init__(self, atom_lib, masses, trajectory_file, bond_file=None, system=None, start_end_nthframe=(0, -1, 1)):
        if "ovito" in sys.modules:
            print("Please remove ovito from loaded modules before using Sample class.")
            sys.exit(1)
        self.trajectory_file = trajectory_file
        self.bond_file = bond_file
        self.system = system
        self.start_frame, self.end_frame, self.nth_frame = start_end_nthframe

        self.sampler_inputs = {"charge_samplers": []}
        self.samplers = []
        self.molecules = {}
        self.bonds = {}

        # Get trajectory data
        with mp.Pool() as pool:
            self.num_particles, self.num_frames, self.frames, self.box = pool.apply_async(self.get_trajectory_data, (trajectory_file, bond_file, self.start_frame, self.end_frame, )).get()
        self.end_frame = self.end_frame if self.end_frame != -1 else self.num_frames - 1
        
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

        # TODO system (+ box)

    def init_from_subprocess(self, atom_lib, masses, trajectory_file, bond_file, system, start_end_nthframe, num_particles, box):
        self.trajectory_file = trajectory_file
        self.bond_file = bond_file
        self.system = system
        self.start_frame, self.end_frame, self.nth_frame = start_end_nthframe

        self.sampler_inputs = {"charge_samplers": []}
        self.samplers = []
        self.molecules = {}
        self.bonds = {}

        self.num_particles = num_particles
        self.num_frames = self.end_frame - self.start_frame + 1
        self.frames = range(self.start_frame, self.end_frame + 1, self.nth_frame)
        self.box = box

        self.type_to_name = {v: k for k, v in atom_lib.items()}
        self.name_to_type = atom_lib
        self.masses = masses
        # TODO system (+ box)

    @staticmethod
    def get_trajectory_data(trajectory_file, bond_file, start_frame, end_frame):
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
            raise ValueError("No bonds found. Ensure bond_file is provided of the trajectory contains bond data.")
        num_frames = pipeline.source.num_frames
        num_particles = first_frame.particles.count
        # Validate start and end frame values
        if start_frame < 0 or end_frame < -1 or (start_frame >= end_frame and end_frame != -1) or end_frame >= num_frames:
            raise ValueError(f"Invalid start_end frame range. The trajectory has {num_frames} frames and the provided range is ({start_frame}, {end_frame}).")
        frames = range(start_frame, end_frame + 1 if end_frame != -1 else num_frames)
        box = np.diagonal(first_frame.cell.matrix)
        return num_particles, num_frames, frames, box

    def add_sampler(self, sampler):
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be an instance of Sampler class.")
        self.samplers.append(sampler)

    def add_charge_sampling(self, link_out, dimension, atoms, num_bins=None, range=None):
        inputs = {"link_out": link_out,
                  "dimension": dimension,
                  "num_bins": num_bins,
                  "range": range,
                  "atoms": atoms}
        # ChargeSampler.validate_inputs(inputs)
        self.sampler_inputs["charge_samplers"].append(inputs)

    def sample(self, is_parallel=True, num_cores=0):
        # Determine number of cores to use
        avail_cores = mp.cpu_count()
        cluster_tasks = (
            os.getenv("SLURM_NTASKS")
            or os.getenv("PBS_NP")
            or os.getenv("LSB_DJOB_NUMPROC")
            or os.getenv("NSLOTS"))
        cluster_tasks = int(cluster_tasks) if cluster_tasks else None
        max_cores = min(avail_cores, cluster_tasks) if cluster_tasks else avail_cores-1
        num_cores = num_cores if num_cores and num_cores<=max_cores else max_cores

        if is_parallel and num_cores > 1:
            start_end_nthframe_list = []
            frames_per_core = (self.end_frame - self.start_frame + 1) // num_cores
            for i in range(num_cores):
                start_frame = self.start_frame + i * frames_per_core
                end_frame = self.start_frame + (i + 1) * frames_per_core - 1 if i < num_cores - 1 else self.end_frame
                start_end_nthframe_list.append((start_frame, end_frame, self.nth_frame))
                print(f"Process {i}: frames {start_frame} to {end_frame}")
            if "ovito" in sys.modules:
                print("Ovito module detected. Please remove it before using parallel sampling. This exit is intentional to infinite loop issues.")
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
                                                                           self.box
                                                                           )) for process_id in range(num_cores)]
                results = []
                pool.close()
                pool.join()
            print([r.get() for r in results])
        else:
            print("Starting serial sampling...")
            self.init_sampler(self.sampler_inputs, process_id=0)
            self.sample_helper()

    @staticmethod
    def init_subprocess_sampler(atom_lib, masses, trajectory_file, bond_file, system, start_end_nthframe, sampler_inputs, process_id, num_particles, box):
        sample_instance = Sample.__new__(Sample)
        sample_instance.init_from_subprocess(atom_lib, masses, trajectory_file, bond_file, system, start_end_nthframe, num_particles=num_particles, box=box)
        sample_instance.init_sampler(sampler_inputs, process_id)
        sample_instance.sample_helper()
        return f"Process {process_id} finished sampling."
    
    def init_sampler(self, sampler_inputs, process_id):
        for sampler in sampler_inputs:
            for params in sampler_inputs[sampler]:
                if sampler == "charge_samplers":
                    self.add_sampler(ChargeSampler(link_out=params["link_out"],
                                                   dimension=params["dimension"],
                                                   atoms=params["atoms"],
                                                   process_id=process_id,
                                                   num_bins=params["num_bins"],
                                                   range=params["range"]))

    def sample_helper(self):
        from ovito.io import import_file
        from ovito.modifiers import LoadTrajectoryModifier
        from ovito.data import BondsEnumerator
        os.environ["OVITO_THREAD_COUNT"] = "1"


        self.pipeline = import_file(self.trajectory_file)
        if self.bond_file:
            bond_modifier = LoadTrajectoryModifier()
            bond_modifier.source.load(self.bond_file)
            self.pipeline.modifiers.append(bond_modifier)

        for sampler in self.samplers:
            if isinstance(sampler, AtomSampler):
                mols = sampler.init_sampling(atom_lib=self.name_to_type)
                self.molecules.update(mols)
            if isinstance(sampler, BondSampler):
                bonds = sampler.init_sampling(self.name_to_type)
                self.bonds.update(bonds)
        
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
        molecule_idx = {identifier: np.zeros(self.num_particles, dtype=int) for identifier in self.molecules}

        # Loop over frames
        for frame_idx in self.frames:
            print(f"Processing frame {frame_idx}...")
            frame = self.pipeline.compute(frame_idx)
            atom_types = frame.particles.particle_types.array
            atom_charges = frame.particles["Charge"].array
            atom_identifiers = frame.particles.identifiers.array
            atom_positions = frame.particles.positions.array
            atom_velocities = frame.particles.velocities.array
            bond_topology = frame.particles.bonds.topology.array
            bond_enum = BondsEnumerator(frame.particles.bonds)

            # Reset molecule indices
            for mol in molecule_idx:
                molecule_idx[mol] = np.zeros(self.num_particles, dtype=int)

            # Identify molecules
            for atom_type in self.type_to_name:
                atoms = np.where(atom_types == atom_type)[0]
                # No molecules registered for this atom type
                if molecules_per_atom_type[atom_type] == []:
                    continue
                # Molecule registered without bond constraints
                elif molecules_per_atom_type[atom_type][0][0] == [[]]:
                    molecule_idx[molecules_per_atom_type[atom_type][0][1]][atoms] = 1
                # No bond constraints, for this atom type
                if len(molecules_per_atom_type[atom_type]) == 1 and molecules_per_atom_type[atom_type][0][0] == [[]]:
                    continue
                # Atom with bond constraints
                for atom in atoms:
                    bonds = list(bond_enum.bonds_of_particle(atom))
                    particles = bond_topology[bonds].flatten()
                    other_particles = particles[particles != atom]
                    other_types = list(atom_types[other_particles])
                    for bond_permutations, indentifier in molecules_per_atom_type[atom_type]:
                        if other_types in bond_permutations:
                            molecule_idx[indentifier][atom] = 1
            for mol in molecule_idx:
                molecule_idx[mol] = np.where(molecule_idx[mol]==1)[0]

            # Sampling
            for sampler in self.samplers:
                if isinstance(sampler, ChargeSampler):
                    sampler.sample(frame=frame_idx,
                                   charges=atom_charges,
                                   mol_index=molecule_idx)
                else:
                    sampler.sample()

        for sampler in self.samplers:
            input_params, data = sampler.get_data()
            data.update({"input_params": input_params})
            file_name = sampler.link_out
            utils.save_object(data, file_name)