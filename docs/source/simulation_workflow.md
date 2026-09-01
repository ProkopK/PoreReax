# Simulation workflow

The `Simulate` class creates LAMMPS-ready inputs from an equilibrated `.gro` structure and writes run scripts for one or more simulation stages.

## Key methods

`set_force_field(...)` Set a custom ReaxFF force-field file path.

`set_job_file(...)` Configure a custom HPC submission template and command.

`add_sim(...)` Add a simulation stage. Common stage types are `nvt`, `npt`, and `nve`.

`add_image_dump(...)` Add optional image-rendering dumps to simulation runs.

`generate()` Write workflow files to the current working directory.

## Generated files

Depending on your setup, generated outputs include:

- `system.data`
- `reax.ffield`
- `run_<n>.lmp`
- `run_<n>.job`
- `ana.py`

## Minimal setup

```python
from porereax import Simulate

gro_lib = {"OM": "O", "OW": "O", "Si": "Si", "HW": "H", "MW": ""}
gro_charges = {"OM": -0.64, "OW": -1.1128, "Si": 1.28, "HW": 0.5564}
atom_masses = {"Si": 28.086, "O": 15.9994, "H": 2.016}
sim = Simulate(gro_lib, gro_charges, atom_masses, structure_file="system.gro")
sim.add_sim(type="nvt", nsteps=10000, temp=300)
sim.generate()
```

## Additional configuration

The simulation workflow can be integrated in an existing PoreSim setup. To do so, you can create a `reax` folder in your PoreSim project and run the script without the `structure_file` argument. The `Simulate` class will then look for a `nvt.gro` file and use it as the starting structure.

To set a custom ReaxFF force-field, use the `set_force_field()` method.

Multiple simulation stages can be added with the `add_sim()` method. The following example shows a typical NVT + NPT workflow:

```python
from porereax import Simulate

# Set up a dictionary to translate GRO atom types to ReaxFF atom types as well as masses and charges
gro_lib = {
    "OM": "O",
    "SI": "Si",
    "Si": "Si",
    "O": "O",
    "H": "H",
    "OW": "O",
    "HW": "H",
    "MW": "",
}
gro_charges = {
    "OM": -0.64,
    "SI": 1.28,
    "Si": 1.28,
    "O": -0.74,
    "H": 0.42,
    "OW": -1.1128,
    "HW": 0.5564,
}
atom_masses = {"Si": 28.086, "O": 15.9994, "H": 2.016}

# Create a simulation object
sim = Simulate(
    gro_lib=gro_lib,
    gro_charges=gro_charges,
    atom_masses=atom_masses,
)

# Set the your own force field
sim.set_force_field(
    force_field="path_to/reax.ffield",
)

# Add simulations
sim.add_sim(
    type="nvt",
    nsteps=10000,
    temp=300,
)

sim.add_sim(
    type="npt",
    nsteps=2000000,
    temp=300,
    pressure=1.0,
    dt=0.5,
    nodes=1,
    tasks_per_node=64,
    wall_time="20:00:00",
    dump_freq=2000,
    thermo_freq=2000,
)

# Generate the simulation files
sim.generate()
```

LAMMPS can create image dumps of the simulation trajectory. To add this feature, use the `add_image_dump()` method. It is designed to visualize pore structures and will cut in the middle of the simulation box.

```python
sim.add_image_dump(
    plane="xy",
    dump_freq=None,
    zoom=3.5,
    image_width=1200,
    image_height=1200,
    atom_sizes={"Si": 3, "O": 2, "H": 1},
    atom_colors={"Si": "orange", "O": "red", "H": "white"},
    # map_by_charge="-1 2 ca 0.0 3 min royalblue 0 green max orangered", # shadows the atom_colors argument
    kwargs="shiny 0.1 box no 0.01",
)
```

Depending on your HPC environment, you may need to set a custom job submission template and command. Use the `set_job_file()` method to configure these options. Use the `lammps_command` argument to specify a custom LAMMPS execution command and use `{input_file}` and `{log_file}` placeholders to indicate where the input and log file names should be inserted. For example:

```python
sim.set_job_file(
    job_file="path_to/job_template.job",
    submit_command="sbatch",
    lammps_command="mpirun --bind-to core --map-by slot lmp -in {input_file} -log {log_file} -k on -sf kk -pk kokkos neigh half newton on comm host",
)
```

The `job_template.job` file should match the requirements of your HPC and can contain placeholders for further customization in the `add_sim()` method. If your using a `$TMPDIR` or similar temporary directory, make shure to copy all required files (`system.data`, `reax.ffield`, etc.) to the temporary directory before running LAMMPS and copy the generated outputs back to the working directory. To sync files like images or `.restart` files, use a background process. The following example shows a job template:

```bash
#!/bin/bash
#SBATCH --nodes={{ SIMULATIONNODES }}
#SBATCH --ntasks-per-node={{ SIMULATIONTASKSPERNODE }}
#SBATCH --ntasks-per-core=1 
#SBATCH --time={{ SIMULATIONTIME }}
#SBATCH --job-name={{ SIMULATIONLABEL }}
#SBATCH --cpus-per-task=1
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
#SBATCH --partition=compute
#sbatch --export=ALL 

# set locale to C
unset LANG
unset LC_CTYPE

# set stack size limit to unlimited:
ulimit -s unlimited

cd "${SLURM_SUBMIT_DIR}"
module purge
module load chem/lammps/22Jul2025_update2

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export OMPI_MCA_btl="^ofi,openib"
export OMPI_MCA_mtl="^ofi"

rsync -avzPu * $TMPDIR/. --include="*.data" --include="*.lmp" --include="*.ffield" --exclude="*"

cd $TMPDIR
mkdir -p figures
(
while true; do
   rsync -azu figures/ ${SLURM_SUBMIT_DIR}/figures/
   rsync -azu --ignore-missing-args *.restart ${SLURM_SUBMIT_DIR}/.
   sleep 300
done
) &
SYNC_RUNNER=$!

{{ LAMMPS_COMMAND }}

rsync -avzPu * ${SLURM_SUBMIT_DIR}/. --include="*.log" --include="*.lammpstrj" --include="*.bonds" --include="*.species" --include="*.data" --exclude="*"
rsync -azu figures/* ${SLURM_SUBMIT_DIR}/figures/.
rsync -azu *.restart ${SLURM_SUBMIT_DIR}/.

kill $SYNC_RUNNER
```