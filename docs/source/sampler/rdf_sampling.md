# RDF sampling

`add_rdf_sampling` computes the radial distribution function g(r) between two registered [atom structures](../analysis/bonding_environment.md).

```python
sampler.add_rdf_sampling(
    name_out="rdf_OSiH_OHH",
    pairs=[
        ({"atom": "O", "bonds": ["Si", "H"]}, {"atom": "O", "bonds": ["H", "H"]}),
    ],
    region="Box",
    num_bins=200,
    r_max=7.0,
)
```

- `pairs`: list of `(atom_A, atom_B)` tuples, each a plain atom structure dictionary. The identifier for a pair joins each side's own atom-structure identifier with `-`, in the order given, e.g. `O(H+Si)-O(H+H)` for the example above - unlike bond identifiers, the two sides here are *not* required to be actually bonded to each other.
- `region`: applied to both `A` and `B` atoms before pairing.
- `num_bins`, `r_max`: histogram resolution and cutoff distance, in Ångstrom.

```{warning}
Increasing `r_max` grows the neighbour search cost; a large cutoff on a large system can make RDF sampling noticeably slower than the other samplers. Start with a modest `r_max` and increase it only as far as you need.
```

<!-- ```{figure} ../images/sampler_rdf_example.png
:alt: Radial distribution function g(r) between two oxygen environments.

Example g(r) for a registered atom pair.
``` -->

## Output

For each identifier, `rdf.obj` stores:

- `num_frames`, `num_atoms_A`, `num_atoms_B`: sampled frame count and total matching atom counts on each side.
- `hist`: the normalised g(r) (`hist -> 1` for large r in a homogeneous system).
- `hist_raw`: the same histogram without the shell-volume/density normalisation (raw average pair count per bin).
- `hist_std`, `bin_edges`: histogram (averaged per frame) and its bin edges.

```{note}
`A == B` pairs (the same atom structure on both sides) are supported for a self-RDF; the neighbour search never pairs an atom with itself, so no special handling is needed for that case.
```
