# Bond length sampling

`add_bond_length_sampling` histograms either the geometric length or the ReaxFF bond order of each registered [bond structure](../analysis/bonding_environment.md).

```python
sampler.add_bond_length_sampling(
    name_out="bond_length_OH",
    bonds=[{"bond": "O-H", "bonds_A": ["H"]}],
    dimension="Bond Length",
    region="Box",
    num_bins=200,
    range=(0.0, 3.0),
)
```

- `bonds`: list of bond structures, see [Bonding environment](../analysis/bonding_environment.md).
- `region`: which part of the box to sample; a bond only counts if **both** its atoms fall inside the region.
- `dimension`: `"Bond Length"` (default) histograms the minimum-image distance between the two atoms in Ångstrom; `"Bond Order"` instead histograms the ReaxFF bond order recorded in the `.bonds` file (dimensionless).
- `range`: `(min, max)` for the histogram - in Ångstrom for `"Bond Length"`, in bond-order units for `"Bond Order"`.

<!-- ```{figure} ../images/sampler_bond_length_example.png
:alt: Bond length histogram for O-H bonds.

Example bond length distribution for a registered bond structure.
``` -->

## Output

For each identifier, `bond_length.obj` stores:

- `num_frames`: number of sampled frames.
- `num_bonds`: total number of matching bonds summed over all frames.
- `mean`: mean bond length (or bond order).
- `hist` / `bin_edges`: histogram (averaged per frame) and its bin edges.
- `std`: standard deviation of the distribution.
- `mean_std`: standard error of the mean.
- `hist_std`: standard deviation of the histogram counts per bin.

```{note}
For `"Bond Order"`, a trajectory/bond file without recorded bond orders is treated as all-zero rather than raising an error, so an empty or flat-at-zero histogram usually means the `.bonds` file doesn't carry that property rather than a real physical result.
```
