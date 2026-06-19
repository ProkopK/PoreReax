"""
Simulation setup and management for ReaxFF molecular dynamics simulations.

This module provides functionality to convert GROMACS structure files (.gro) to
LAMMPS data files and generate complete simulation workflows for ReaxFF force field
calculations. It handles atom type mapping, charge assignment, and creates all
necessary input files for running molecular dynamics simulations on HPC systems.

The module automatically generates:
- LAMMPS data files from GROMACS structures
- LAMMPS input scripts for equilibration and production runs
- Job submission scripts for HPC clusters
- Analysis scripts for post-processing simulation results

Example
-------
>>> from porereax.simulate import Simulate
>>> gro_lib = {"Si": "Si", "O": "O", "OM": "O", "HW": "H", "MW": ""}
>>> gro_charges = {'Si': 2.4, 'O': -1.2, 'H': 0.6, 'OM': -0.8}
>>> atom_masses = {'Si': 28.085, 'O': 15.999, 'H': 1.008}
>>> sim = Simulate(gro_lib, gro_charges, atom_masses, 'system.gro')
>>> sim.set_force_field('reax.ffield')
>>> sim.add_sim('nvt', nsteps=100000, temp=300)
>>> sim.generate()
"""

import os
import shutil

from jinja2 import Template


class Simulate():
    """
    Main class for setting up and managing ReaxFF molecular dynamics simulations.

    This class handles the conversion of GROMACS structure files to LAMMPS format,
    manages atom type mappings and charges, and generates all necessary files for
    running ReaxFF simulations on HPC systems.

    Example
    -------
    >>> gro_lib = {"Si": "Si", "O": "O", "OM": "O", "HW": "H", "MW": ""}
    >>> gro_charges = {'Si': 2.4, 'O': -1.2, 'H': 0.6, 'OM': -0.8}
    >>> atom_masses = {'Si': 28.085, 'O': 15.999, 'H': 1.008}
    >>> sim = Simulate(gro_lib, gro_charges, atom_masses, 'system.gro')
    """

    def __init__(self, gro_lib, gro_charges, atom_masses, structure_file=None):
        """
        Initialize the Simulate object with atom mappings and structure file.

        Parameters
        ----------
        gro_lib : dict
            Dictionary mapping GROMACS atom names (keys) to ReaxFF atom type names (values).
            Use empty string "" for virtual sites or atoms to exclude from ReaxFF simulation.
            Example: {"Si": "Si", "O": "O", "OM": "O", "HW": "H", "MW": ""}
        gro_charges : dict
            Dictionary mapping GROMACS atom names (keys) to partial charges (values) in
            elementary charge units. Must include charges for all atoms in gro_lib except
            those mapped to empty string.
            Example: {'Si': 2.4, 'O': -1.2, 'H': 0.6, 'OM': -0.8}
        atom_masses : dict
            Dictionary mapping ReaxFF atom type names (keys) to atomic masses (values) in
            atomic mass units (amu). Must include masses for all atom types used in gro_lib.
            Example: {'Si': 28.085, 'O': 15.999, 'H': 1.008}
        structure_file : str, optional
            Path to the GROMACS structure file (.gro) to use as input. If None, the method
            will look for 'nvt.gro' in the parent directory's 'nvt' folder. Default is None.

        Raises
        ------
        FileNotFoundError
            If the specified structure file does not exist.
        ValueError
            If gro_lib, gro_charges, or atom_masses are not properly formatted dictionaries,
            or if there are missing mappings between these dictionaries.

        Notes
        -----
        The constructor performs extensive validation to ensure all required mappings are
        present and correctly formatted. Virtual sites or excluded atoms should be mapped
        to empty string "" in gro_lib.
        """
        self._path = os.getcwd()
        if structure_file is None:
            self._name = os.path.basename(os.path.split(self._path)[0])
            self._structure_file = os.path.join(os.path.split(self._path)[0], "nvt", "nvt.gro")
        else:
            self._name = os.path.basename(self._path)
            self._structure_file = os.path.abspath(structure_file)
        if not os.path.isfile(self._structure_file):
            raise FileNotFoundError(f"Structure file {self._structure_file} not found.")

        # Validate gro_lib
        if not isinstance(gro_lib, dict):
            raise ValueError("gro_lib must be provided as a dictionary mapping atom names to atom types.")
        if not all(isinstance(k, str) for k in gro_lib.keys()):
            raise ValueError("All keys in gro_lib must be strings representing atom names.")
        if not all(isinstance(v, str) for v in gro_lib.values()):
            raise ValueError("All values in gro_lib must be strings representing atom names in the reaxFF force field.")
        self._gro_lib = gro_lib
        self._type_to_name = dict(enumerate(set([v for v in gro_lib.values() if v != ""]), start=1))
        self._name_to_type = {v: k for k, v in self._type_to_name.items()}
        self._num_atom_types = max(self._type_to_name.keys())

        # Validate gro_charges
        if not isinstance(gro_charges, dict):
            raise ValueError("gro_charges must be provided as a dictionary mapping atom names to charges.")
        if not all(isinstance(k, str) for k in gro_charges.keys()):
            raise ValueError("All keys in gro_charges must be strings representing atom names.")
        if not all(isinstance(v, (int, float)) for v in gro_charges.values()):
            raise ValueError("All values in gro_charges must be numbers representing charges.")
        if not all(gro_charges.get(k) is not None for k in gro_lib.keys() if gro_lib[k] != ""):
            raise ValueError("All atom names in gro_lib (except those mapped to \"\") must have corresponding charges in gro_charges.")
        self._gro_charges = gro_charges

        # Validate atom_masses
        if not isinstance(atom_masses, dict):
            raise ValueError("atom_masses must be provided as a dictionary mapping atom types to masses.")
        if not all(isinstance(k, str) for k in atom_masses.keys()):
            raise ValueError("All keys in atom_masses must be strings representing atom names.")
        if not all(isinstance(v, (int, float)) and v > 0 for v in atom_masses.values()):
            raise ValueError("All values in atom_masses must be positive numbers representing masses.")
        if not all(atom_masses.get(v) is not None for v in self._type_to_name.values()):
            raise ValueError("All atoms in gro_lib must have corresponding masses in atom_masses.")
        self.atom_masses = atom_masses

        self._job_file = None
        self._submit_cmd = None
        self._lammps_command = None
        self._force_field = None
        self._sim = []

    def set_job_file(self, file_path, submit_command, lammps_command=None):
        """
        Specify a custom job submission template file and command.

        This method allows you to provide a custom job script template for HPC job
        submission. The template should use Jinja2 syntax and include placeholders
        for job parameters like node count, wall time, and LAMMPS commands.

        Parameters
        ----------
        file_path : str
            Path to the job submission template file. Must be a valid file that exists.
            The template should be compatible with the HPC scheduler (SLURM, PBS, etc.).
        submit_command : str
            Command to submit jobs to the scheduler (e.g., "sbatch" for SLURM,
            "qsub" for PBS/Torque). Must be a non-empty string.
        lammps_command : str, optional
            Custom command to run LAMMPS. If None, a default MPI command will be used.
            Use placeholders {input_file} and {log_file} for input and log file names.

        Raises
        ------
        FileNotFoundError
            If the specified job file does not exist.
        ValueError
            If submit_command is not a non-empty string.

        Notes
        -----
        If this method is not called, a default SLURM template will be used with
        'sbatch' as the submission command.
        If lammps_command is not provided, a default MPI command will be used.

        Example
        -------
        >>> sim.set_job_file('/path/to/custom.job', 'sbatch', 'mpirun lmp -in {input_file} -log {log_file}')
        """
        # Validate and set job file path
        if file_path is None and self._job_file is not None:
            f_path = os.path.join(os.path.dirname(__file__), "templates", "reax.job")
        else:
            f_path = os.path.abspath(file_path)

        # Validate and set submit command
        if submit_command is None and self._submit_cmd is not None:
            s_command = "sbatch"
        else:
            if not isinstance(submit_command, str) or submit_command == "":
                raise ValueError("submit_command must be a non-empty string.")
            s_command = submit_command.strip()

        # Validate and set LAMMPS command
        if lammps_command is None and self._lammps_command is not None:
            l_command = "mpirun lmp -in {input_file} -log {log_file} -k on -sf kk -pk kokkos neigh half newton on comm host"
        else:
            if not isinstance(lammps_command, str) or lammps_command == "":
                raise ValueError("lammps_command must be a non-empty string.")
            l_command = lammps_command.strip()

        if not os.path.isfile(f_path):
            raise FileNotFoundError(f"Job file {f_path} not found.")
        self._job_file = f_path
        self._submit_cmd = s_command
        self._lammps_command = l_command

    def set_force_field(self, force_field):
        """
        Specify a custom ReaxFF force field parameter file.

        Parameters
        ----------
        force_field : str
            Path to the ReaxFF force field parameter file (ffield). This file contains
            all the reactive force field parameters for the atom types in the system.

        Raises
        ------
        FileNotFoundError
            If the specified force field file does not exist.

        Notes
        -----
        If this method is not called, a default Si/O/H force field from
        https://doi.org/10.1063/1.3407433 will be used.

        Example
        -------
        >>> sim.set_force_field('/path/to/reax.ffield')
        """
        if force_field is None:
            ffield = os.path.join(os.path.dirname(__file__), "templates", "reax.ffield")
            print("Using force field from https://doi.org/10.1063/1.3407433 for Si/O/H systems.")
        else:
            ffield = os.path.abspath(force_field)

        if not os.path.isfile(ffield):
            raise FileNotFoundError(f"Force field file {ffield} not found.")
        self._force_field = ffield

    def add_sim(self, type, nsteps, temp, pressure=1.0, dt=0.5, nodes=1, tasks_per_node=64, wall_time="20:00:00", dump_freq=100, thermo_freq=100):
        """
        Add a simulation step to the workflow.

        This method adds a molecular dynamics simulation step with specified parameters.
        Multiple simulation steps can be added sequentially to create a multi-stage
        workflow (e.g., equilibration followed by production runs).

        Parameters
        ----------
        type : str
            Ensemble type for the simulation. Common values:
            - 'nvt': Constant number of particles, volume, and temperature
            - 'npt': Constant number of particles, pressure, and temperature
            - 'nve': Constant number of particles, volume, and energy
        nsteps : int
            Number of MD steps to run in this simulation stage.
        temp : float
            Temperature in Kelvin for the simulation.
        pressure : float, optional
            Pressure in atmospheres for NPT simulations. Default is 1 atm.
        dt : float, optional
            Time step in femtoseconds. Default is 0.5 fs.
        nodes : int, optional
            Number of compute nodes to request for this job. Default is 1.
        tasks_per_node : int, optional
            Number of MPI tasks per node. Default is 64.
        wall_time : str, optional
            Maximum wall time for the job in HH:MM:SS format. Default is "20:00:00".
        dump_freq : int, optional
            Frequency (in steps) for writing trajectory snapshots. Default is 100.
        thermo_freq : int, optional
            Frequency (in steps) for writing thermodynamic output. Default is 100.

        Notes
        -----
        Simulation steps are executed in the order they are added. Each step will
        automatically submit the next step upon completion if multiple steps exist.

        Example
        -------
        >>> # Add equilibration run
        >>> sim.add_sim('npt', nsteps=500000, temp=300, pressure=1, dt=0.5)
        >>> # Add production run
        >>> sim.add_sim('nvt', nsteps=50000, temp=300, dt=0.5)
        """
        self._sim.append({
            "type": type,
            "nsteps": nsteps,
            "temp": temp,
            "pressure": pressure,
            "dt": dt,
            "thermo_freq": thermo_freq,
            "dump_freq": dump_freq,
            "nodes": nodes,
            "tasks_per_node": tasks_per_node,
            "wall_time": wall_time,
        })

    def generate(self):
        """
        Generate all simulation files and scripts.

        This method is the main entry point for generating a complete simulation workflow.
        It creates all necessary files for running ReaxFF simulations:

        - Converts GROMACS structure to LAMMPS data file (system.data)
        - Copies or uses default ReaxFF force field file (reax.ffield)
        - Generates LAMMPS input scripts for initial equilibration and all simulation steps
        - Creates job submission scripts for each simulation stage
        - Generates analysis script (ana.py) for post-processing

        The method automatically chains job submissions so that each simulation step
        submits the next one upon completion.

        Raises
        ------
        ValueError
            If atom names in the structure file are not found in gro_lib or if charges
            are missing from gro_charges.

        Notes
        -----
        - If no force field is specified, uses default Si/O/H parameters
        - If no job template is specified, uses default SLURM template
        - Virtual sites and excluded atoms (mapped to "" in gro_lib) are filtered out
        - Prints detailed information about system composition and total charge

        Example
        -------
        >>> sim = Simulate(gro_lib, gro_charges, atom_masses, 'system.gro')
        >>> sim.set_force_field('reax.ffield')
        >>> sim.add_sim('nvt', nsteps=100000, temp=300)
        >>> sim.generate()
        Using force field from https://doi.org/10.1063/1.3407433 for Si/O/H systems.
        Box dimensions (Angstroms): [50.0, 50.0, 50.0]
        Total charge in system: 0.0000e
        Atom counts by type:
          Type Si: 100 atoms
          Type O: 200 atoms
        LAMMPS data file written to /path/to/system.data
        """
        # Create and convert gro to lammps data
        self._create_input_file(os.path.join(self._path, "system.data"))

        # Setup and copy force field file
        if self._force_field is None:
            self.set_force_field(None)
        shutil.copy2(self._force_field, os.path.join(self._path, 'reax.ffield'))

        if self._job_file is None or self._submit_cmd is None or self._lammps_command is None:
            self.set_job_file(None, None, None)

        with open(self._job_file, 'r') as f:
            job_template = Template(f.read())
        with open(os.path.join(os.path.dirname(__file__), "templates", "run_n.lmp"), 'r') as f:
            lmp_step_template = Template(f.read())

        atoms = ' '.join(self._type_to_name[k] for k in range(1, self._num_atom_types + 1))

        for step_idx, step in enumerate(self._sim):
            file_name = f"run_{step_idx}"
            lmp_file = os.path.join(self._path, f"{file_name}.lmp")
            with open(lmp_file, 'w') as f:
                file_content = lmp_step_template.render(
                    SIMNUMBER=step_idx,
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
            job_file = os.path.join(self._path, f"{file_name}.job")
            with open(job_file, 'w') as f:
                file_content = job_template.render(
                    SIMULATIONNODES=step["nodes"],
                    SIMULATIONTASKSPERNODE=step["tasks_per_node"],
                    SIMULATIONTIME=step["wall_time"],
                    SIMULATIONLABEL=f"{self._name}_run_{step_idx}",
                    LAMMPS_COMMAND=self._lammps_command.format(
                        input_file=f"{file_name}.lmp",
                        log_file=f"{file_name}.log"
                    ),
                )
                f.write(file_content)
                if step_idx+1<len(self._sim):
                    f.write(f"\n\n{self._submit_cmd} run_{step_idx+1}.job\n")

        # Create ana files
        with open(os.path.join(os.path.dirname(__file__), "templates", "ana.py"), 'r') as f:
            ana_template = Template(f.read())
        with open(os.path.join(self._path, "ana.py"), 'w') as f:
            file_content = ana_template.render(
                NUMSIMS=len(self._sim),
                NAME_TO_TYPE=self._name_to_type,
                ATOM_MASSES=self.atom_masses,
            )
            f.write(file_content)

    def _line_mapper(self, line):
        """
        Parse a line from a GROMACS structure file and extract atom information.

        This internal method parses the fixed-width format of GROMACS .gro files and
        converts units from GROMACS conventions (nm, nm/ps) to LAMMPS conventions
        (Angstroms, Angstroms/ps). Also removes numeric suffixes from atom names.

        Parameters
        ----------
        line : str
            A single line from the atom section of a .gro file. Expected format:
            - Columns 1-5: Residue ID
            - Columns 6-10: Residue name
            - Columns 11-15: Atom name
            - Columns 21-28: X coordinate (nm)
            - Columns 29-36: Y coordinate (nm)
            - Columns 37-44: Z coordinate (nm)
            - Columns 45-52: X velocity (nm/ps)
            - Columns 53-60: Y velocity (nm/ps)
            - Columns 61-68: Z velocity (nm/ps)

        Returns
        -------
        tuple
            A tuple containing:
            - res_id (int): Residue ID number
            - res_name (str): Residue name
            - atom_name (str): Atom name with numeric suffixes removed
            - x (float): X coordinate in Angstroms
            - y (float): Y coordinate in Angstroms
            - z (float): Z coordinate in Angstroms
            - vx (float): X velocity in Angstroms/ps
            - vy (float): Y velocity in Angstroms/ps
            - vz (float): Z velocity in Angstroms/ps

        Notes
        -----
        - Coordinates are converted from nm to Angstroms (multiply by 10)
        - Velocities are converted from nm/ps to Angstroms/ps (multiply by 10,
          but then divide by 1000 for ps to fs, net division by 100)
        - Numeric suffixes are stripped from atom names (e.g., 'O1' becomes 'O')
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

    def _read_gro_file(self):
        """
        Read and parse a GROMACS structure file.

        This internal method reads the structure file, extracts simulation box dimensions,
        and parses all atom information including positions and velocities.

        Returns
        -------
        tuple
            A tuple containing:
            - box_dims (list): List of box dimensions [x, y, z] in Angstroms.
              Converted from nm in the .gro file.
            - gro_data (list): List of tuples, where each tuple contains parsed
              atom information (res_id, res_name, atom_name, x, y, z, vx, vy, vz)
              as returned by _line_mapper method.

        Notes
        -----
        - Box dimensions are read from the last line of the .gro file
        - Atom data is read from lines 3 to second-to-last (skipping header and box line)
        - Prints box dimensions to stdout for user verification
        """
        with open(self._structure_file, 'r') as file:
            lines = [l for l in file.readlines() if l.strip()]

        box_dims = [float(dim) * 10 for dim in lines[-1].strip().split()]
        print(f"Box dimensions (Angstroms): [{', '.join(f'{dim:.3f}' for dim in box_dims)}]")

        gro_data = [self._line_mapper(line) for line in lines[2:-1]]

        return box_dims, gro_data

    def _write_lammps_header(self, file, num_atoms, box_dims):
        """
        Write the header section of a LAMMPS data file.

        This internal method writes the header information including atom counts,
        atom type counts, simulation box dimensions, and atomic masses. The format
        follows LAMMPS data file conventions.

        Parameters
        ----------
        file : file object
            Open file object in write mode where the header will be written.
        num_atoms : int
            Total number of atoms in the system (excluding virtual sites).
        box_dims : list
            List of box dimensions [x, y, z] in Angstroms. Assumes orthorhombic box
            with lower bounds at 0.0.

        Notes
        -----
        The header includes:
        - Title line
        - Atom and atom type counts
        - Box boundaries (xlo xhi, ylo yhi, zlo zhi)
        - Mass section with masses for each atom type
        """
        file.write("System generated by PoreReax package\n\n")
        file.write(f"{num_atoms} atoms\n")
        file.write(f"{self._num_atom_types} atom types\n\n")
        file.write(f"{0.0:8.3f} {box_dims[0]:8.3f} xlo xhi\n")
        file.write(f"{0.0:8.3f} {box_dims[1]:8.3f} ylo yhi\n")
        file.write(f"{0.0:8.3f} {box_dims[2]:8.3f} zlo zhi\n\n")
        file.write("Masses\n\n")
        for atom_name, mass in self.atom_masses.items():
            file.write(f"{self._name_to_type[atom_name]} {mass:8.3f}\n")

    def _write_lammps_data(self, file, gro_data):
        """
        Write the Atoms and Velocities sections of a LAMMPS data file.

        This internal method writes the atomic coordinates, charges, and velocities
        to the LAMMPS data file. It validates that all atoms have proper type and
        charge assignments and calculates total system charge and atom type counts.

        Parameters
        ----------
        file : file object
            Open file object in write mode where atom data will be written.
        gro_data : list
            List of tuples with atom data as returned by _read_gro_file method.
            Each tuple contains: (res_id, res_name, atom_name, x, y, z, vx, vy, vz).

        Raises
        ------
        ValueError
            If an atom name is not found in gro_lib or if charge information is
            missing from gro_charges for any atom.

        Notes
        -----
        - Uses 'charge' atom style format: atom-ID molecule-ID atom-type q x y z
        - Prints total system charge and atom counts by type to stdout
        - Charge neutrality should be verified by the user from printed output
        """
        charge_count = 0.0
        atom_count = {atom_type: 0 for atom_type in range(1, self._num_atom_types + 1)}
        file.write("\nAtoms\n\n")
        for i, data in enumerate(gro_data):
            res_id, _, atom_name, x, y, z, _, _, _ = data
            atom_type = self._name_to_type.get(self._gro_lib.get(atom_name))
            charge = self._gro_charges.get(atom_name)
            if atom_type is None:
                raise ValueError(f"Atom name '{atom_name}' not found in gro_lib.")
            if charge is None:
                raise ValueError(f"Charge for atom name '{atom_name}' not found in gro_charges.")
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
            print(f"  Type {self._type_to_name[atom_type]}: {count} atoms")

    def _create_input_file(self, out_path):
        """
        Create a complete LAMMPS data file from GROMACS structure file.

        This internal method orchestrates the reading of the GROMACS structure file,
        filtering of virtual sites, and writing of the complete LAMMPS data file
        including header, masses, atom coordinates, and velocities.

        Parameters
        ----------
        out_path : str
            Path where the output LAMMPS data file should be written.

        Notes
        -----
        - Automatically filters out atoms mapped to empty string "" in gro_lib
          (e.g., virtual sites from TIP4P water model)
        - Prints box dimensions, total charge, and atom counts to stdout
        - The created file is in LAMMPS 'charge' atom style format
        """
        box_dims, gro_data = self._read_gro_file()
        gro_data = [data for data in gro_data if self._gro_lib.get(data[2], 0) != ""] # Filter out ghost particles e.g. from tip4p water model
        num_atoms = len(gro_data)
        with open(out_path, 'w') as file:
            self._write_lammps_header(file, num_atoms, box_dims)
            self._write_lammps_data(file, gro_data)
        print(f"LAMMPS data file written to {out_path}")