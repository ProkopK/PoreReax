# Analysis

This section covers the `Sample` class and everything around post-processing a trajectory.

- [Analysis workflow](analysis_workflow.md) explains how to set up a `Sample` instance, registers samplers, and runs the sampling.
- [Bonding environments](bonding_environment.md) explains how atoms and bonds are matched to local structures, and how PoreReax builds the identifiers used to name those.
- [Regions](regions.md) documents the built-in and custom `region=` filters shared by every sampler.
- [Visualisation](visualization.md) shows how to load and plot the `.obj` files each sampler produces.

The individual samplers (charge, density, angle, ...) are documented separately in the [sampler reference](../sampler/index.md).

```{toctree}
:hidden:

analysis_workflow
bonding_environment
regions
visualization
```
