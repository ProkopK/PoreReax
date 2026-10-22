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

```{include} ../../README.md
:start-after: <!-- start minimal simulation workflow -->
:end-before: <!-- end minimal simulation workflow -->
```

(additional-configuration)=
## Additional configuration

The simulation workflow can be integrated in an existing [PoreSim](https://github.com/PoreMS/PoreSim) setup. To do so, you can create a `reax` folder in your PoreSim project and run the script without the `structure_file` argument. The `Simulate` class will then look for a `nvt.gro` and `pore.yml` file simulation directory and use it as the starting structure and pore geometry.

To set a custom ReaxFF force-field, use the `set_force_field()` method.

Multiple simulation stages can be added with multiple `add_sim()` methods. The following example shows a typical NVT + NpT workflow:

```python
from porereax import Simulate

# Set up a dictionary to translate GRO atom types to ReaxFF
# atom types as well as masses and charges
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

# Set the used force field
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

## Adding image dumps

LAMMPS can create image dumps of the simulation trajectory. To add this feature, use the `add_image_dump()` method. It is designed to visualise pore structures and will cut in the middle of the simulation box, in the given plane.

```python
sim.add_image_dump(
    plane="yz",
    dump_freq=None,  # use the dump frequency of the simulation stage
    zoom=3.5,
    image_width=1200,
    image_height=1200,
    atom_sizes={"Si": 3, "O": 2, "H": 1},
    atom_colors={"Si": "orange", "O": "red", "H": "white"},
    # map_by_charge="-1 2 ca 0.0 3 min royalblue 0 green max orangered", # shadows the atom_colors argument
    kwargs="shiny 0.1 box no 0.01",
)
```

```{figure} images/lammps_img_dump_rotated.png
:alt: Example LAMMPS-rendered image dump of a pore cross-section.

Example rendered image dump, cutting through the yz plane of a cylindrical pore filled with water.
```

## Freezing atoms

For pore/confinement systems it might be desirable to freeze certain atoms to prevent a drift of the solid wall. `freeze_region()` duplicates the atom types of a given region into "frozen" counterparts, fixed in LAMMPS with `fix setforce 0 0 0`. For systems created with [PoreMS](https://github.com/PoreMS/PoreMS) this region can be derived automatically from the `pore.yml` file with `auto_freeze()`.

```python
# Freeze everything outside a 10 Angstrom reactive shell around a cylindrical pore
sim.auto_freeze(reactive_radius=10.0)

# Or freeze a custom region directly
sim.freeze_region(lambda x, y, z: z < 5.0)
```

```{note}
`auto_freeze()` requires a `pore.yml` file (i.e. no `structure_file=` was given) and currently only supports cylindrical pores.
```

## HPC job setup

Depending on your HPC environment, you may need to set a custom job submission template and command. Use the `set_job_file()` method to configure these options. Use the `lammps_command` argument to specify a custom LAMMPS execution command and use `{input_file}` and `{log_file}` placeholders to indicate where the input and log file names should be inserted. For example:

```python
sim.set_job_file(
    job_file="path_to/job_template.job",
    submit_command="sbatch",
    lammps_command="mpirun --bind-to core --map-by slot lmp -in {input_file} -log {log_file} -k on -sf kk -pk kokkos neigh half newton on comm host",
)
```

The `job_template.job` file should match the requirements of your HPC and can contain placeholders for further customization in the `add_sim()` method. If your using a `$TMPDIR` or similar temporary directory, make sure to copy all required files (`system.data`, `reax.ffield`, etc.) to the temporary directory before running LAMMPS and copy the generated outputs back to the working directory. To sync files like images or `.restart` files, use a background process. The following example shows an example job template:

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
