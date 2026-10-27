# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 

### Fixed

- 

## [0.1.0] - 2026-10-27

First public release.

### Added

- `Simulate`: build LAMMPS data files and multi-stage ReaxFF workflows (job
  scripts, image dumps, frozen regions) from GROMACS structures, including
  PoreSim/PoreMS integration.
- `Sample`: bonding-environment based analysis of LAMMPS trajectories with
  molecule structure, charge, angle, bond length, density, bond density, RDF and
  reaction samplers, in serial or parallel.
- `plot_data`, `get_data`, `get_identifiers` and `load_object` for plotting and
  inspecting sampler results.
- Documentation at <https://prokopk.github.io/PoreReax/>.

[0.1.0]: https://github.com/ProkopK/PoreReax/releases/tag/v0.1.0
