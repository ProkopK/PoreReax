# Getting started

## Requirements

- Python 3.12 or newer
- LAMMPS for running generated workflows
- ReaxFF parameter file for your target chemistry (or the provided default Si/O/H file)

## Installation

Direct installation from github:

```bash
python -m pip install git+https://github.com/ProkopK/PoreReax.git
```

## Core concepts

`Simulate` Builds simulation files from a GROMACS `.gro` structure.

`Sample` Loads trajectory/bond data and runs one or more analysis samplers.

`plot(...)` Visualizes sampled results.

## Typical workflow

1. Create simulation files with `Simulate` and run LAMMPS.
2. Load generated outputs with `Sample`.
3. Add desired samplers (charge, density, angle, bond, reaction, RDF, etc.).
4. Run sampling and visualize the results.