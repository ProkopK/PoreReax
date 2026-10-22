# Reaction sampling

`add_reaction_sampling` counts bond-formation/bond-breaking events, defined as an atom whose bonding environment changes from a "reactant" [atom structure](../analysis/bonding_environment.md) to a "product" one between two consecutive frames. It reuses the same positional/time histogramming as [density sampling](density_sampling.md).

```python
sampler.add_reaction_sampling(
    name_out="reaction_SiOH_break",
    reactions=[
        ({"atom": "O", "bonds": ["Si", "H"]}, {"atom": "O", "bonds": ["Si"]}),
    ],
    dimension="Time",
    position="center",
)
```

- `reactions`: list of `(reactant, product)` tuples. Each side is a normal atom structure dictionary, or `None` to mean "any atom, regardless of type or environment" (a wildcard for the whole side). The identifier for a reaction joins the reactant and product identifiers with `-`, using `X` for a `None` side, e.g. `O(H+Si)-O(Si)` or `X-O(Si+Si)`.
- `dimension`/`direction`/`num_bins`: same options as [density sampling](density_sampling.md).
- `position`: which position to record a reaction event at:
  - `"center"`: the minimum-image midpoint between the atom's position in the previous and current frame.
  - `"reactant"`: the atom's position in the previous frame.
  - `"product"`: the atom's position in the current frame.

## How a reaction is detected

An atom counts as a reaction event for a given `(reactant, product)` pair only if **both** are true for that frame transition:

1. At least one of its bonds was actually formed or broken (a real topology change). Reactions are detected from the discrete bond topology already assigned by the ReaxFF bond file.
2. It matched the `reactant` structure in the *previous* frame and the `product` structure in the *current* frame.

<!-- ```{figure} ../images/sampler_reaction_example.png
:alt: Time series of Si-OH bond breaking events.

Example reaction-event density over time.
``` -->

## Output

Same fields as the [density sampling output](density_sampling.md): `num_frames`/`densities` for `"Time"`, or `num_frames`/`hist`/`hist_std`/`bin_edges` (`x_edges`/`y_edges` for 2D) for the histogram dimensions - one entry per `(reactant, product)` identifier.

```{note}
This definition of a "reaction" is influenced by "Bond Flickering", where a bond is close to the bond-order cutoff and may appear to form and break repeatedly over a few frames. 
```
