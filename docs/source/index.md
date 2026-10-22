---
hide-toc: true
---

# PoreReax

```{include} ../../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

(typical-workflow)=
## Typical workflow

1. [Create simulation](simulation_workflow.md) files with `Simulate` and run LAMMPS.
2. [Load generated outputs](analysis/index.md) with `Sample`.
3. Add desired [samplers](sampler/index.md) (charge, density, angle, bond, reaction, RDF, etc.).
4. Configure [bonding environments](analysis/bonding_environment.md) and structures of interest.
5. Run sampling in parallel.
6. [Visualise](analysis/visualization.md) the results.

```{toctree}
:hidden:

getting_started
simulation_workflow
analysis/index
sampler/index
```

```{toctree}
:caption: API
:hidden:

api
```
