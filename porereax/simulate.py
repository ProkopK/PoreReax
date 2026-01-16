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
>>> sim.set_force_field('ffield.reax')
>>> sim.add_sim('nvt', nsteps=100000, temp=300)
>>> sim.generate()
"""

import os

from jinja2 import Template


class Simulate():
    """
    Main class for setting up and managing ReaxFF molecular dynamics simulations.

    This class handles the conversion of GROMACS structure files to LAMMPS format,
    manages atom type mappings and charges, and generates all necessary files for
    running ReaxFF simulations on HPC systems.

    Attributes
    ----------
    path : str
        Current working directory for simulation files.
    name : str
        Name of the simulation system.
    structure_file : str
        Path to the input GROMACS structure file (.gro).
    gro_lib : dict
        Mapping from GROMACS atom names to ReaxFF atom types.
    type_to_name : dict
        Mapping from numeric atom type IDs to ReaxFF atom type names.
    name_to_type : dict
        Mapping from ReaxFF atom type names to numeric IDs.
    num_atom_types : int
        Total number of unique atom types in the system.
    gro_charges : dict
        Mapping from atom names to their partial charges.
    atom_masses : dict
        Mapping from atom type names to their masses in atomic mass units.
    job_file : str or None
        Path to the job submission template file.
    submit_cmd : str or None
        Command to submit jobs (e.g., 'sbatch', 'qsub').
    lamps_command : str or None
        Custom command to run LAMMPS.
    force_field : str or None
        Path to the ReaxFF force field parameter file.
    sim : list
        List of simulation steps to execute.

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
        self.path = os.getcwd()
        if structure_file is None:
            self.name = os.path.basename(os.path.split(self.path)[0])
            self.structure_file = os.path.join(os.path.split(self.path)[0], "nvt", "nvt.gro")
        else:
            self.name = os.path.basename(self.path)
            self.structure_file = structure_file
        if not os.path.isfile(self.structure_file):
            raise FileNotFoundError(f"Structure file {self.structure_file} not found.")

        # Validate gro_lib
        if not isinstance(gro_lib, dict):
            raise ValueError("gro_lib must be provided as a dictionary mapping atom names to atom types.")
        if not all(isinstance(k, str) for k in gro_lib.keys()):
            raise ValueError("All keys in gro_lib must be strings representing atom names.")
        if not all(isinstance(v, str) for v in gro_lib.values()):
            raise ValueError("All values in gro_lib must be strings representing atom names in the reaxFF force field.")
        self.gro_lib = gro_lib
        self.type_to_name = dict(enumerate(set([v for v in gro_lib.values() if v != ""]), start=1))
        self.name_to_type = {v: k for k, v in self.type_to_name.items()}
        self.num_atom_types = max(self.type_to_name.keys())

        # Validate gro_charges
        if not isinstance(gro_charges, dict):
            raise ValueError("gro_charges must be provided as a dictionary mapping atom names to charges.")
        if not all(isinstance(k, str) for k in gro_charges.keys()):
            raise ValueError("All keys in gro_charges must be strings representing atom names.")
        if not all(isinstance(v, (int, float)) for v in gro_charges.values()):
            raise ValueError("All values in gro_charges must be numbers representing charges.")
        if not all(gro_charges.get(k) is not None for k in gro_lib.keys() if gro_lib[k] != ""):
            raise ValueError("All atom names in gro_lib (except those mapped to \"\") must have corresponding charges in gro_charges.")
        self.gro_charges = gro_charges

        # Validate atom_masses
        if not isinstance(atom_masses, dict):
            raise ValueError("atom_masses must be provided as a dictionary mapping atom types to masses.")
        if not all(isinstance(k, str) for k in atom_masses.keys()):
            raise ValueError("All keys in atom_masses must be strings representing atom names.")
        if not all(isinstance(v, (int, float)) and v > 0 for v in atom_masses.values()):
            raise ValueError("All values in atom_masses must be positive numbers representing masses.")
        if not all(atom_masses.get(v) is not None for v in self.type_to_name.values()):
            raise ValueError("All atoms in gro_lib must have corresponding masses in atom_masses.")
        self.atom_masses = atom_masses

        self.job_file = None
        self.submit_cmd = None
        self.lamps_command = None
        self.force_field = None
        self.force_field_atoms = None
        self.sim = []

    def set_job_file(self, file_path, sumbit_command, lammps_command=None):
        """
        Specify a custom job submission template file and command.

        This method allows you to provide a custom job script template for HPC job
        submission. The template should use Jinja2 syntax and include placeholders
        for job parameters like node count, wall time, and LAMMPS commands.

        Parameters
        ----------
        file_path : str
            Path to the job submission template file. Must be a valid file that exists.
            The template should be compatible with your HPC scheduler (SLURM, PBS, etc.).
        sumbit_command : str
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
            If sumbit_command is not a non-empty string.

        Notes
        -----
        If this method is not called, a default SLURM template will be used with
        'sbatch' as the submission command.
        If lammps_command is not provided, a default MPI command will be used.

        Example
        -------
        >>> sim.set_job_file('/path/to/custom.job', 'sbatch', 'mpirun lmp -in {input_file} -log {log_file}')
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Job file {file_path} not found.")
        if not isinstance(sumbit_command, str) or sumbit_command == "":
            raise ValueError("submit_cmd must be a not empty string.")
        self.submit_cmd = sumbit_command
        self.lamps_command = lammps_command
        self.job_file = file_path

    def set_force_field(self, force_field):
        """
        Specify a custom ReaxFF force field parameter file.

        Parameters
        ----------
        force_field : str
            Path to the ReaxFF force field parameter file (ffield). This file contains
            all the reactive force field parameters for the atom types in your system.

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
        >>> sim.set_force_field('/path/to/ffield.reax')
        """
        if not os.path.isfile(force_field):
            raise FileNotFoundError(f"Force field file {force_field} not found.")
        self.force_field = force_field

    def add_sim(self, type, nsteps, temp, pressure=1.0, dt=0.5, nodes=1, taskes_per_node=64, wall_time="20:00:00", dump_freq=100, thermo_freq=100):
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
        taskes_per_node : int, optional
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
        >>> sim.add_sim('nvt', nsteps=50000, temp=300, dt=0.5)
        >>> # Add production run
        >>> sim.add_sim('npt', nsteps=500000, temp=300, pressure=1, dt=0.5)
        """
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
        >>> sim.set_force_field('ffield.reax')
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
        self._create_input_file(os.path.join(self.path, "system.data"))

        # Copy force field file
        if self.force_field is None:
            self.force_field = os.path.join(os.path.dirname(__file__), "templates", "reax.ffield")
            print("Using force field from https://doi.org/10.1063/1.3407433 for Si/O/H systems.")
        os.system(f"cp {self.force_field} {os.path.join(self.path, 'reax.ffield')}")

        if self.job_file is None:
            self.job_file = os.path.join(os.path.dirname(__file__), "templates", "reax.job")
            self.submit_cmd = "sbatch"
        if self.lamps_command is None:
            self.lamps_command = "mpirun lmp -in {input_file} -log {log_file} -k on -sf kk -pk kokkos neigh half newton on comm host"

        with open(self.job_file, 'r') as f:
            job_template = Template(f.read())
        with open(os.path.join(os.path.dirname(__file__), "templates", "reax_run_n.lmp"), 'r') as f:
            lmp_step_template = Template(f.read())

        atoms = ' '.join(self.type_to_name[k] for k in range(1, self.num_atom_types + 1))

        for step_idx, step in enumerate(self.sim):
            file_name = f"reax_run_{step_idx}"
            lmp_file = os.path.join(self.path, f"{file_name}.lmp")
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
            job_file = os.path.join(self.path, f"{file_name}.job")
            with open(job_file, 'w') as f:
                file_content = job_template.render(
                    SIMULATIONNODES=step["nodes"],
                    SIMULATIONTASKSPERNODE=step["tasks_per_node"],
                    SIMULATIONTIME=step["wall_time"],
                    SIMULATIONLABEL=f"{self.name}_run_{step_idx}",
                    LAMMPS_COMMAND=self.lamps_command.format(
                        input_file=f"{file_name}.lmp",
                        log_file=f"{file_name}.log"
                    ),
                )
                f.write(file_content)
                if step_idx<len(self.sim):
                    f.write(f"\n\n{self.submit_cmd} reax_run_{step_idx}.job\n")

        # Create ana files
        with open(os.path.join(os.path.dirname(__file__), "templates", "ana.py"), 'r') as f:
            ana_template = Template(f.read())
        with open(os.path.join(self.path, "ana.py"), 'w') as f:
            file_content = ana_template.render(
                NUMSIMS=len(self.sim),
                NAME_TO_TYPE=self.name_to_type,
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
        with open(self.structure_file, 'r') as file:
            lines = file.readlines()

        box_dims = [float(dim) * 10 for dim in lines[-1].strip().split()]
        print("Box dimensions (Angstroms):", box_dims)

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
        file.write(f"{self.num_atom_types} atom types\n\n")
        file.write(f"{0.0:8.3f} {box_dims[0]:8.3f} xlo xhi\n")
        file.write(f"{0.0:8.3f} {box_dims[1]:8.3f} ylo yhi\n")
        file.write(f"{0.0:8.3f} {box_dims[2]:8.3f} zlo zhi\n\n")
        file.write("Masses\n\n")
        for atom_name, mass in self.atom_masses.items():
            file.write(f"{self.name_to_type[atom_name]} {mass:8.3f}\n")

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
        atom_count = {atom_type: 0 for atom_type in range(1, self.num_atom_types + 1)}
        file.write("\nAtoms\n\n")
        for i, data in enumerate(gro_data):
            res_id, _, atom_name, x, y, z, _, _, _ = data
            atom_type = self.name_to_type.get(self.gro_lib.get(atom_name))
            charge = self.gro_charges.get(atom_name)
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
            print(f"  Type {self.type_to_name[atom_type]}: {count} atoms")

    def _create_input_file(self, file_path):
        """
        Create a complete LAMMPS data file from GROMACS structure file.

        This internal method orchestrates the reading of the GROMACS structure file,
        filtering of virtual sites, and writing of the complete LAMMPS data file
        including header, masses, atom coordinates, and velocities.

        Parameters
        ----------
        file_path : str
            Path where the LAMMPS data file should be written.

        Notes
        -----
        - Automatically filters out atoms mapped to empty string "" in gro_lib
          (e.g., virtual sites from TIP4P water model)
        - Prints box dimensions, total charge, and atom counts to stdout
        - The created file is in LAMMPS 'charge' atom style format
        """
        box_dims, gro_data = self._read_gro_file()
        gro_data = [data for data in gro_data if self.gro_lib.get(data[2], 0) != ""] # Filter out ghost particles e.g. from tip4p water model
        num_atoms = len(gro_data)
        with open(file_path, 'w') as file:
            self._write_lammps_header(file, num_atoms, box_dims)
            self._write_lammps_data(file, gro_data)
        print(f"LAMMPS data file written to {file_path}")