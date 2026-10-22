# Molecule structure sampling

`add_molecule_structure_sampling` takes an inventory of every bonding environment that actually occurs in the trajectory, for every atom type - without you having to specify which structures to look for. It is usually the first sampler to run on a new system: it tells you which structures exist at all, before you set up targeted [charge](charge_sampling.md), [angle](angle_sampling.md), or [density](density_sampling.md) sampling for the ones you care about.

```python
sampler.add_molecule_structure_sampling(
    name_out="molecule_structure", region="Box", steps=1
)
```

- `region`: which part of the box to count structures in, see [Regions](../analysis/regions.md).
- `steps`: how many bonding "steps" out from the central atom to describe. `steps=1` (the default) only records the directly bonded neighbours, e.g. `O(Si+Si)`. `steps=2` also expands each of those neighbours' own neighbours, e.g. `O(Si(O+O+O)+Si(O+O+O))`, and so on for larger values. Larger `steps` values distinguish more structures but gets computationally expensive and grow the number of unique identifiers quickly.

Identifiers are built the same way as described in [Bonding environment](../analysis/bonding_environment.md) - `Atom(neighbour+neighbour(...))`, alphabetically-ish sorted and `+`-joined - except they are *discovered* automatically rather than declared, so no `atoms=[...]` argument is needed.

<!-- ```{figure} ../images/sampler_molecule_structure_example.png
:alt: Bar chart of average structure counts per frame for one atom type.

Example molecule structure output: average count per frame for each discovered bonding environment of one atom type.
``` -->

## Output

Unlike every other sampler, `molecule_structure.obj` is not keyed by structure identifier at the top level; it is one dictionary per **atom type name**, each mapping that atom type's discovered identifiers to their average count per frame, sorted from most to least common:

```python
{
    "O": {"O(Si+Si)": 812.3, "O(H+Si)": 140.1, "O(H+H)": 64.0, ...},
    "Si": {"Si(O+O+O+O)": 501.7, ...},
    "H": {"H(O)": 205.4, ...},
    "input_params": {...},
}
```

`get_identifiers("molecule_structure.obj")` therefore returns atom type names (`"O"`, `"Si"`, `"H"`), and `get_data(..., "O")` returns that atom type's whole structure-count dictionary. [`plot_data`](../analysis/visualization.md) plots the first identifier of the identifiers list as a bar chart.
