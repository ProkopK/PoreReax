################################################################################
# Simulate                                                                     #
#                                                                              #
"""Module to convert GROMACS .gro files to LAMMPS data files."""               #
################################################################################

import numpy as np
import os
import sys
from jinja2 import Template, Environment, FileSystemLoader


class Simulate():
    def __init__(self, gro_lib, gro_charges, atom_masses, atom_lib, poreSim_structure=True, structure_file=None):
        self.path = os.getcwd()
        self.name = os.path.basename(self.path)
        if poreSim_structure:
            self.structure_file = os.path.join(self.path, "nvt", "nvt.gro")
        else:
            self.structure_file = structure_file
        if self.structure_file is None:
            raise ValueError("Structure file must be provided if poreSim_structure is False.")
        if not os.path.isfile(self.structure_file):
            raise FileNotFoundError(f"Structure file {self.structure_file} not found.")

        # Validate gro_lib
        if not isinstance(gro_lib, dict):
            raise ValueError("gro_lib must be provided as a dictionary mapping atom names to atom types.")
        if not all(isinstance(k, str) for k in gro_lib.keys()):
            raise ValueError("All keys in gro_lib must be strings representing atom names.")
        if not all(isinstance(v, int) and (v >= 1 or v == -1) for v in gro_lib.values()):
            raise ValueError("All values in gro_lib must be integers >= 1 representing atom types or -1 for ghost particles.")
        self.gro_lib = gro_lib
        self.num_atom_types = max(gro_lib.values())

        # Validate gro_charges
        if not isinstance(gro_charges, dict):
            raise ValueError("gro_charges must be provided as a dictionary mapping atom names to charges.")
        if not all(isinstance(k, str) for k in gro_charges.keys()):
            raise ValueError("All keys in gro_charges must be strings representing atom names.")
        if not all(isinstance(v, (int, float)) for v in gro_charges.values()):
            raise ValueError("All values in gro_charges must be numbers representing charges.")
        if not all(gro_charges.get(k, None) is not None for k in gro_lib.keys() if gro_lib[k] != -1):
            raise ValueError("All atom names in gro_lib (except those mapped to -1) must have corresponding charges in gro_charges.")
        self.gro_charges = gro_charges

        # Validate atom_masses
        if not isinstance(atom_masses, dict):
            raise ValueError("atom_masses must be provided as a dictionary mapping atom types to masses.")
        if not all(isinstance(k, int) and k >= 1 for k in atom_masses.keys()):
            raise ValueError("All keys in atom_masses must be integers >= 1 representing atom types.")
        if not all(isinstance(v, (int, float)) and v > 0 for v in atom_masses.values()):
            raise ValueError("All values in atom_masses must be positive numbers representing masses.")
        if not all(atom_masses.get(k, None) is not None for k in range(1, self.num_atom_types + 1)):
            raise ValueError("All atom types in gro_lib must have corresponding masses in atom_masses.")
        self.atom_masses = atom_masses

        # Validate atom_lib
        if not isinstance(atom_lib, dict):
            raise ValueError("atom_lib must be provided as a dictionary mapping atom types to atom names.")
        if not all(isinstance(k, int) and k >= 1 for k in atom_lib.keys()):
            raise ValueError("All keys in atom_lib must be integers >= 1 representing atom types.")
        if not all(isinstance(v, str) for v in atom_lib.values()):
            raise ValueError("All values in atom_lib must be strings representing atom names.")
        if not all(atom_lib.get(k, None) is not None for k in range(1, self.num_atom_types + 1)):
            raise ValueError("All atom types in gro_lib must have corresponding names in atom_lib.")
        self.atom_lib = atom_lib

        self.job_file = None
        self.force_field = None
        self.force_field_atoms = None
        self.sim = []

    def add_job_file(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Job file {file_path} not found.")
        self.job_file = file_path

    def add_force_field(self, force_field):
        if not os.path.isfile(force_field):
            raise FileNotFoundError(f"Force field file {force_field} not found.")
        self.force_field = force_field

    def add_sim(self, type, nsteps, temp, pressure=1, dt=0.5, nodes=1, taskes_per_node=64, wall_time="20:00:00", dump_freq=100, thermo_freq=100):
        self.sim.append({
            "type": type,
            "nsteps": nsteps,
            "temp": temp,
            "pressure": pressure,
            "dt": dt,
            "thermo_freq": thermo_freq,
            "dump_freq": dump_freq,
            "nodes": nodes,
            "tasks_per_node": taskes_per_node,
            "wall_time": wall_time,
        })

    def generate(self):
        # Create ReaxFF directory
        os.makedirs(os.path.join(self.path, "reax"), exist_ok=True)

        self.__create_input_file(os.path.join(self.path, "reax", "system.data"))

        if self.force_field is None:
            self.force_field = os.path.join(os.path.dirname(__file__), "templates", "reax.ffield")
            print("Using force field from https://doi.org/10.1063/1.3407433 for Si/O/H systems.")
        # Copy force field file
        os.system(f"cp {self.force_field} {os.path.join(self.path, 'reax', 'reax.ffield')}")

        if self.job_file is None:
            self.job_file = os.path.join(os.path.dirname(__file__), "templates", "reax.job")
        with open(self.job_file, 'r') as f:
            job_template = Template(f.read())

        with open(os.path.join(os.path.dirname(__file__), "templates", "reax_run_0.lmp"), 'r') as lf:
            lmp_initial_template = Template(lf.read())

        with open(os.path.join(os.path.dirname(__file__), "templates", "reax_run_n.lmp"), 'r') as lf:
            lmp_step_template = Template(lf.read())

        atoms = ' '.join(self.atom_lib[k] for k in range(1, self.num_atom_types + 1))

        # Create initial LAMMPS input file and shell scripts
        with open(os.path.join(self.path, "reax", "reax_run_0.lmp"), 'w') as f:
            file_content = lmp_initial_template.render(
                atoms=atoms
            )
            f.write(file_content)
        # Create job file for initial run
        with open(os.path.join(self.path, "reax", "reax_run_0.job"), 'w') as f:
            file_content = job_template.render(
                SIMULATIONNODES=1,
                SIMULATIONPROCS=10,
                SIMULATIONTIME="0:30:00",
                SIMULATIONLABEL=f"{self.name}_initial",
                FILENAME="reax_run_0"
            )
            f.write(file_content)
            if self.sim:
                f.write(f"\n\nsbatch reax_run_1.job\n")

        for step_idx, step in enumerate(self.sim):
            file_name = f"reax_run_{step_idx+1}"
            lmp_file = os.path.join(self.path, "reax", f"{file_name}.lmp")
            with open(lmp_file, 'w') as f:
                file_content = lmp_step_template.render(
                    SIMNUMBER=step_idx + 1,
                    TIMESTEP=step["dt"],
                    THERMO_FREQ=step["thermo_freq"],
                    DUMP_FREQ=step["dump_freq"],
                    ATOMS=atoms,
                    SIMULATIONTYPE=step["type"],
                    TEMP=step["temp"],
                    PRESS=step["pressure"],
                    NSTEPS=step["nsteps"]
                )
                f.write(file_content)
            job_file = os.path.join(self.path, "reax", f"{file_name}.job")
            with open(job_file, 'w') as f:
                file_content = job_template.render(
                    SIMULATIONNODES=step["nodes"],
                    SIMULATIONPROCS=step["tasks_per_node"],
                    SIMULATIONTIME=["wall_time"],
                    SIMULATIONLABEL=f"{self.name}_run_{step_idx+1}",
                    FILENAME=f"reax_run_{step_idx+1}",
                )
                f.write(file_content)
                if step_idx+1<len(self.sim):
                    f.write(f"\n\nsbatch reax_run_{step_idx+1}.job\n")
                
        # Create ana files

    def __line_mapper(self, line):
        """
        Map a line from a .gro file to its components.
        
        Parameters
        ----------
        line : str
            A line from a .gro file.

        Returns
        -------
        tuple
            A tuple containing:
            - res_id (int): Residue ID
            - res_name (str): Residue name
            - atom_name (str): Atom name
            - x (float): X coordinate in Angstroms
            - y (float): Y coordinate in Angstroms
            - z (float): Z coordinate in Angstroms
            - vx (float): X velocity in Angstroms/ps
            - vy (float): Y velocity in Angstroms/ps
            - vz (float): Z velocity in Angstroms/ps
        """
        res_id = int(line[0:5].strip())
        res_name = line[5:10].strip()
        atom_name = line[10:15].strip().translate(str.maketrans("", "", "0123456789"))
        x = float(line[20:28].strip()) * 10
        y = float(line[28:36].strip()) * 10
        z = float(line[36:44].strip()) * 10
        vx = float(line[44:52].strip()) / 100
        vy = float(line[52:60].strip()) / 100
        vz = float(line[60:68].strip()) / 100
        return res_id, res_name, atom_name, x, y, z, vx, vy, vz

    def __read_gro_file(self):
        """

        Returns
        -------
        tuple
            A tuple containing:
            - box_dims (list): List of box dimensions [x, y, z] in Angstroms
            - gro_data (list): List of tuples with atom data
        """
        with open(self.structure_file, 'r') as file:
            lines = file.readlines()
            
        box_dims = [float(dim) * 10 for dim in lines[-1].strip().split()]
        print("Box dimensions (Angstroms):", box_dims)
        
        gro_data = [self.__line_mapper(line) for line in lines[2:-1]]

        return box_dims, gro_data

    def __write_lammps_header(self, file, num_atoms, box_dims):
        """
        Write the header section of a LAMMPS data file.
        
        Parameters
        ----------
        file : file object
            The file object to write to.
        num_atoms : int
            Number of atoms.
        box_dims : list
            List of box dimensions [x, y, z] in Angstroms.
        """
        file.write("System generated by PoreReax package\n\n")
        file.write(f"{num_atoms} atoms\n")
        file.write(f"{self.num_atom_types} atom types\n\n")
        file.write(f"{0.0:8.3f} {box_dims[0]:8.3f} xlo xhi\n")
        file.write(f"{0.0:8.3f} {box_dims[1]:8.3f} ylo yhi\n")
        file.write(f"{0.0:8.3f} {box_dims[2]:8.3f} zlo zhi\n\n")
        file.write("Masses\n\n")
        for atom_type, mass in self.atom_masses.items():
            file.write(f"{atom_type} {mass:8.3f}\n")

    def __write_lammps_data(self, file, gro_data):
        """
        Write the atom and velocity sections of a LAMMPS data file.

        Parameters
        ----------
        file : file object
            The file object to write to.
        gro_data : list
            List of tuples with atom data.
        """
        charge_count = 0.0
        atom_count = {atom_type: 0 for atom_type in range(1, self.num_atom_types + 1)}
        file.write("\nAtoms\n\n")
        for i, data in enumerate(gro_data):
            res_id, _, atom_name, x, y, z, _, _, _ = data
            atom_type = self.gro_lib.get(atom_name)
            charge = self.gro_charges.get(atom_name)
            if atom_type is None:
                raise ValueError(f"Atom name '{atom_name}' not found in atom library.")
            if charge is None:
                raise ValueError(f"Charge for atom name '{atom_name}' with type '{atom_type}' not found in atom charges.")
            charge_count += charge
            atom_count[atom_type] += 1
            file.write(f"{i+1:5d} {res_id:5d} {atom_type:5d} {charge:8.4f} {x:8.3f} {y:8.3f} {z:8.3f}\n")
        file.write("\nVelocities\n\n")
        for i, data in enumerate(gro_data):
            _, _, _, _, _, _, vx, vy, vz = data
            file.write(f"{i+1:5d} {vx: 2.6f} {vy: 2.6f} {vz: 2.6f}\n")

        print(f"Total charge in system: {charge_count:.4f}e")
        print("Atom counts by type:")
        for atom_type, count in atom_count.items():
            print(f"  Type {atom_type}: {count} atoms")

    def __create_input_file(self, file_path):
        """

        Parameters
        ----------
        file_path : str
            Path to the output LAMMPS data file.
        """
        box_dims, gro_data = self.__read_gro_file()
        gro_data = [data for data in gro_data if self.gro_lib.get(data[2], 0) != -1] # Filter out ghost particles e.g. from tip4p water model     
        num_atoms = len(gro_data)
        with open(file_path, 'w') as file:
            self.__write_lammps_header(file, num_atoms, box_dims)
            self.__write_lammps_data(file, gro_data)
        print(f"LAMMPS data file written to {file_path}")