import numpy as np
from ovito.io import import_file
from ovito.modifiers import LoadTrajectoryModifier
from ovito.data import BondsEnumerator

import porereax.utils as utils
from porereax.charge import ChargeSampler
from porereax.utils import Sampler, AtomSampler, BondSampler


class Sample:
    def __init__(self, atom_lib, masses, trajectory_file, bond_file=None, system=None, start_end=(0, -1)):
        self.trajectory_file = trajectory_file
        self.bond_file = bond_file
        self.system = system
        self.start_frame, self.end_frame = start_end

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

        # Load trajectory
        if not isinstance(trajectory_file, str):
            raise TypeError("trajectory_file must be a string path to the trajectory file.")
        self.pipeline = import_file(self.trajectory_file)
        if self.bond_file:
            if not isinstance(bond_file, str):
                raise TypeError("bond_file must be a string path to the bond file.")
            bond_modifier = LoadTrajectoryModifier()
            bond_modifier.source.load(self.bond_file)
            self.pipeline.modifiers.append(bond_modifier)

        # Get and validate trajectory meta data
        self.first_frame = self.pipeline.compute()
        if self.first_frame.particles.count == 0:
            raise ValueError("No particles found in the trajectory file.")
        if self.first_frame.particles.bonds is None:
            raise ValueError("No bonds found. Ensure bond_file is provided of the trajectory contains bond data.")
        self.num_frames = self.pipeline.source.num_frames
        self.num_particles = self.first_frame.particles.count
        # Validate start and end frame values
        if self.start_frame < 0 or self.end_frame < -1 or (self.start_frame >= self.end_frame and self.end_frame != -1) or self.end_frame >= self.num_frames:
            raise ValueError(f"Invalid start_end frame range. The trajectory has {self.num_frames} frames and the provided range is ({self.start_frame}, {self.end_frame}).")
        self.frames = range(self.start_frame, self.end_frame + 1 if self.end_frame != -1 else self.num_frames)
        self.box = np.diagonal(self.first_frame.cell.matrix)

        # TODO system (+ box)

    def add_sampler(self, sampler):
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be an instance of Sampler class.")
        self.samplers.append(sampler)

    def _register_molecule(self, molecule, bonds=None):
        pass

    def sample(self):
        for sampler in self.samplers:
            if isinstance(sampler, AtomSampler):
                mols = sampler.init_sampling(self.name_to_type)
                self.molecules.update(mols)
            if isinstance(sampler, BondSampler):
                bonds = sampler.init_sampling(self.name_to_type)
                self.bonds.update(bonds)
        print(f"Mols: {self.molecules}")

        for frame_idx in self.frames:
            frame = self.pipeline.compute(frame_idx)
            atom_types = frame.particles.particle_types.array
            atom_charges = frame.particles["Charge"].array
            atom_identifiers = frame.particles.identifiers.array
            atom_positions = frame.particles.positions.array
            atom_velocities = frame.particles.velocities.array
            bond_topology = frame.particles.bonds.topology.array
            bond_enum = BondsEnumerator(frame.particles.bonds)
            for sampler in self.samplers:
                if isinstance(sampler, ChargeSampler):
                    sampler.sample(frame=frame_idx,
                                   charges=atom_charges,
                                   types=atom_types,
                                   bond_topology=bond_topology,
                                   bond_enum=bond_enum)
                else:
                    sampler.sample()

        for sampler in self.samplers:
            input_params, data = sampler.get_data()
            data.update({"input_params": input_params})
            file_name = sampler.file_name
            utils.save_object(data, file_name)