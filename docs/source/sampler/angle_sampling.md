# Angle sampling

`add_angle_sampling` histograms the angle formed by the central atom and its bonded neighbours.

```python
sampler.add_angle_sampling(
    name_out="angle_Si-O-Si",
    atoms=[{"atom": "O", "bonds": ["Si", "Si"]}],
    region="Box",
    num_bins=180,
    angle="all",
)
```

- `atoms`: list of [atom structures](../analysis/bonding_environment.md).
- `region`: which part of the box to sample, see [Regions](../analysis/regions.md).
- `num_bins`: number of histogram bins over the fixed `(0°, 180°)` range.
- `angle`: either `"all"`, or `"A-B-C"` with concrete atom identifiers.

  - `"all"` (default): every pair of neighbours contributes an angle - e.g. for an `C(C+H+H+H)`, all combinations of two neighbours are counted, so the three H-C-H angles and the three C-C-H angles are all included.
  - `"a-b-c"`: only count the angle formed with one neighbour of type `a` and one of type `c` (its middle atom `b` must match the central atom type of the registered `atoms` structures); other neighbours of `B` are ignored - e.g. for an `C(C+H+H+H)` with `angle="C-C-H"`, only the three C-C-H angles are counted. The order of `a` and `c` does not matter, so `angle="H-C-C"` is equivalent to `angle="C-C-H"`.

```{important}
An atom structure without a `"bonds"` key, or with fewer than two entries in `"bonds"`, is silently dropped - an angle needs at least two bonded neighbours to be defined at all. Always give `atoms` structures with an explicit `"bonds"` list of at least two atoms (nested-environment entries count the same as plain strings).
```

<!-- ```{figure} ../images/sampler_angle_example.png
:alt: Angle histogram for an Si-O-Si bridging oxygen.

Example angle distribution for a registered atom structure.
``` -->

## Output

For each identifier, `angle.obj` stores:

- `num_frames`: number of frames with at least one matching angle.
- `num_angles`: total number of angles summed over all frames.
- `mean`: mean angle in degrees.
- `hist` / `bin_edges`: angle histogram (averaged per frame, in degrees) and its bin edges.
- `std`: standard deviation of the distribution.
- `mean_std`: standard error of the mean.
- `hist_std`: standard deviation of the histogram counts per bin.
