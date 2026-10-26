
# Sampler

This section documents each sampler you can register on a `Sample` instance:
what it measures, which arguments it takes, and what its `.obj` output
contains. The arguments shared by every sampler - `name_out`, `region` and the
bonding-environment `atoms`/`bonds` specifications - are described once in the
[bonding environment](../analysis/bonding_environment.md).

- [Molecule structure](molecule_structure_sampling.md) discovers the bonding
  environments present in the system, instead of taking a declared list.
- [Charge](charge_sampling.md), [Angle](angle_sampling.md) and
  [Bond length](bond_length_sampling.md) histogram a per-atom or per-bond
  quantity.
- [Density](density_sampling.md) resolves atoms, bonds and structures in space
  or over time.
- [RDF](rdf_sampling.md) computes radial distribution functions between two
  bonding environments.
- [Reaction](reaction_sampling.md) counts bond-topology changes between frames.

```{toctree}
:hidden:

molecule_structure_sampling
charge_sampling
angle_sampling
density_sampling
bond_length_sampling
rdf_sampling
reaction_sampling
```
