# Density sampling

"Density" here covers *how many* of a given [atom structure](../analysis/bonding_environment.md) or [bond structure](../analysis/bonding_environment.md) are present, either over time or as a function of position in the box. `add_density_sampling` counts atoms; its sibling `add_bond_density_sampling` counts bonds using the same dimensions, direction options, and output layout - this page documents the shared machinery once, and the bond variant page only calls out what's different.

```python
sampler.add_density_sampling(
    name_out="density_water_z",
    atoms=[{"atom": "O", "bonds": ["H", "H"]}],
    dimension="Cartesian1D",
    region="Box",
    num_bins=200,
    direction="z",
)
sampler.add_bond_density_sampling(
    name_out="density_OH_z",
    bonds=[{"bond": "O-H", "bonds_A": ["H"]}],
    dimension="Time",
)
```

## Dimensions

`dimension` selects what is being histogrammed:

| `dimension` | What it does |
|---|---|
| `"Time"` | Number of matching atoms per frame - a time series, not a histogram. |
| `"Cartesian1D"` | 1D positional histogram along one Cartesian axis of the simulation box. |
| `"Cartesian2D"` | 2D positional histogram over one Cartesian plane of the simulation box. |
| `"Pore1D"` | Same as `"Cartesian1D"`, but in pore-relative coordinates. Requires `Sample(..., system=...)`. |
| `"Pore2D"` | Same as `"Cartesian2D"`, but in pore-relative coordinates. Requires `Sample(..., system=...)`. |

`direction` then picks the axis/plane, depending on `dimension` (unused for `"Time"`):

| `dimension` | `direction` options |
|---|---|
| `"Cartesian1D"` | `"x"`, `"y"`, `"z"` |
| `"Cartesian2D"` | `"xy"`, `"xz"`, `"yz"` |
| `"Pore1D"` | `"z"` (pore axis), `"r"` (radial distance), `"p"` (azimuthal angle φ), `"d"` (signed distance from the nearest pore mouth) |
| `"Pore2D"` | `"rp"`, `"rz"`, `"pz"` |

`num_bins` sets the number of histogram bins (for both axis, in the 2D case); it is ignored for `"Time"`.

## Conditions

`conditions` further filters which atoms or bonds count, evaluated per frame in addition to `region` and the bonding-environment match:

- `"Charge"`: `(min, max)` tuple - only count atoms whose ReaxFF charge falls in this range.
- `"Angle"`: `(min, max)` tuple - only count atoms that have *at least one* angle (formed with any two of their bonded neighbours) falling in this range.
- `"Bond Length"`: `(min, max)` tuple in Ångstrom, filtering by the length of the bond

```python
sampler.add_density_sampling(
    name_out="density_protonated_O",
    atoms=[{"atom": "O", "bonds": ["H"]}],
    dimension="Time",
    conditions={"Charge": (-0.9, -0.5)},
)
```

<!-- ```{figure} ../images/sampler_density_example.png
:alt: 1D density profile of water oxygens along z.

Example atom density profile along one Cartesian direction.
``` -->

## Output

For `"Time"`, each identifier stores `num_frames` and `densities` (count per sampled frame). For the histogram dimensions, each identifier stores `num_frames`, `hist` (counts per frame, averaged), `hist_std`, and `bin_edges` (`x_edges`/`y_edges` for the 2D case) plus `direction`. Position histograms are **not** normalized to a physical density by the sampler itself - [`plot_data(..., density=True)`](../analysis/visualization.md) converts a `"Cartesian1D"`/`"Pore1D"`/`"Cartesian2D"`/`"Pore2D"` histogram to particles per nm³.
