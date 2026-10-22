# Regions

Every `add_*_sampling` method accepts a `region=` argument that restricts sampling to part of the simulation box. It selects the position based on the central atom (or bond midpoint) of a bonding environment. It can either be used with a custom function or one of the built-in region names.

```python
sampler.add_charge_sampling(
    name_out="charge_pore",
    atoms=[{"atom": "O", "bonds": ["H", "H"]}],
    region="Box",
)
```

## Built-in regions

- `"Box"` (default): the whole simulation box. Always available, no `system=` required.
- `"Reservoir"`: the bulk reservoir outside the pore. Takes only the solvent infront of the wall, excluding the solvent at the entry of the pore.
- `"Pore"`: the inner part of the pore, excluding the reservoir and an entry margin of 3 Å. Does not cut out the wall.
- `"Wall"` / `"Wall_<thickness>"`: a shell of the given `thickness` (default 5 Å) around the surface of the pore, consists of reservoir and pore inside.

```{note}
The built-in regions `"Reservoir"`, `"Pore"`, and `"Wall"` require a `system="pore.yml"`, as created by [PoreMS](https://github.com/PoreMS/PoreMS), so the exact pore geometry is known.
```


## Custom regions

A user defined region can be applied by passing a function to the `region=` argument. It must accept an `(N, 3)` array of positions in Ångstrom and return an `(N,)` boolean array indicating which positions are inside the region. 

```python
# Only the lower half of the box along z
sampler.add_density_sampling(
    name_out="density_lower_half",
    atoms=[{"atom": "O"}],
    dimension="Time",
    region=lambda coords: coords[:, 2] < box_z / 2,
)
```
