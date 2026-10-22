# Charge sampling

`add_charge_sampling` histograms the partial (ReaxFF) charge of the central atom in each registered [atom structure](../analysis/bonding_environment.md).

```python
sampler.add_charge_sampling(
    name_out="charge_water_O",
    atoms=[{"atom": "O", "bonds": ["H", "H"]}],
    region="Box",
    range=(-2.0, 2.0),
)
```

- `atoms`: list of atom structures to sample, see [Bonding environment](../analysis/bonding_environment.md).
- `region`: which part of the box to sample, see [Regions](../analysis/regions.md).
- `range`: `(min, max)` charge range in elementary charge *e* for the histogram. Internally the sampler bins at 0.001*e* resolution (charges are rounded to 3 decimal places), so the number of bins is fixed by the range you choose, not separately configurable.

<!-- ```{figure} ../images/sampler_charge_example.png
:alt: Charge histogram for a water oxygen.

Example charge distribution for a registered atom structure.
``` -->

## Output

For each identifier, `charge.obj` stores:

- `num_frames`: number of sampled frames.
- `num_atoms`: total number of matching atoms summed over all frames.
- `mean`: mean charge over all matching atoms.
- `hist` / `bin_edges`: charge histogram (averaged per frame) and its bin edges.
- `std`: standard deviation of the distribution.
- `mean_std`: standard error of the mean.
- `hist_std`: standard deviation of the histogram counts per bin.

```{note}
If your trajectory is missing charges, or OVITO is not able to read them, charges are silently treated as `0.0` rather than raising an error.
```