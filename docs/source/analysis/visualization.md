# Visualisation

Every sampler writes its results to a `<name_out>.obj` file: a pickled dictionary keyed by the [identifiers](bonding_environment.md) of the atoms/bonds/pairs/reactions you registered, plus an `input_params` entry describing how the sampler was configured (dimension, region, bin settings, ...).

## Loading raw data

```python
from porereax import load_object, get_identifiers, get_data

data = load_object("charge.obj")
identifiers = get_identifiers("charge.obj")
charge_H2O = get_data("charge.obj", "O(H+H)")
```

- `load_object` returns the full dictionary, including `input_params`.
- `get_identifiers` lists only the data identifiers present in the file, without any data or `input_params`.
- `get_data` looks up a single identifier and returns all its data.

## Plotting with `plot_data`

```python
from porereax import plot_data
import matplotlib.pyplot as plt

fig, ax = plot_data("charge.obj")
plt.show()
```

Useful keyword arguments for the 1D/time case:

- `identifiers`: which identifiers to plot (default: all of them).
- `colors`: override the default color cycle.
- `mean=True`: draw a vertical line at the sampled mean (charge/angle/bond length only).
- `std=True`: shade ±1 standard deviation around the histogram (and the mean, if also enabled).
- `density=True`: normalise histograms to reasonable units, e.g. charge per atom, density per nm³
- `dt`: timestep in femtoseconds, for `dimension == "Time"` plots.
- `plot_kwargs_1d`: dict of extra keyword arguments passed to `Axes.plot` for 1D/time plots.

For 2D plots:
- `transpose=True` swaps the two axes
- `plot_kwargs_2d` dict of extra keyword arguments passed to `Axis.pcolormesh`

Pass an existing `axis=` to draw into a subplot you already created instead of creating a new figure.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plot_data("molecule_structure.obj", axis=axes[0])
plot_data("angle_Si-O-Si.obj", axis=axes[1])
plot_data("density_water_2d_yz.obj", axis=axes[2])
```

```{figure} ../images/quickstart_image.png
:alt: A figure combining a molecule-structure bar chart, a 1D histogram line plot, and a 2D density heatmap.

Combining several `.obj` files into one figure with `plot_data(..., axis=...)`.
```

```{note}
For most work the default `plot_data` can be used, for further customization you can load the raw data and plot it yourself.
```
