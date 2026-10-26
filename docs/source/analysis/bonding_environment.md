# Bonding environment

The PoreReax analysis is not based on whole molecules or species, but on the **bonding environment** of each atom and bond. It defines local structures in terms of the central atom (or bond) and the types of atoms it is bonded to. This allows to analyse substructures of larger structures, by searching for the specific bonding environment in the bonding graph. For example a geminal silanol group can be described as a silicon atom bonded to four oxygen, with two of them having an hydrogen bonded to them.

These bonding environments are defined by the user as Python dictionaries, and then outputted by the samplers as text identifiers.

## Atom structures

An atom structure is a dictionary with:

- `"atom"`: the atom identifier (a key of the `atom_lib`), the central atom of the structure.
- `"bonds"` (optional): a list describing the atoms this central atom must be bonded to.

```python
{"atom": "O", "bonds": ["H", "H"]}
```

A few things to note about the `"bonds"` list:
- If **no `"bonds"` key** is specified, it allows any structure with that central atom.
- IF **`"bonds": []`** is given, it describes structures with no further bonds.
- Each entry can be a **wildcard** `"X"` to match any atom type in the system.
- Any `"bonds"` entry can be an atom dictonary itself, to describe the neighbour's own bonding environment. In the example above of a geminal silanol would be described as:

```python
{
    "atom": "Si",
    "bonds": [
        "O",
        "O",
        {"atom": "O", "bonds": ["H"]},
        {"atom": "O", "bonds": ["H"]},
    ],
}
```

## Bond structures

A bond structure is a dictionary with:

- `"bond"`: two atom identifiers separated by `-`, e.g. `"O-H"`
- `"bonds_A"` / `"bonds_B"` (optional): lists describing the *additional* neighbours of the first / second atom in the bond. These lists work the same way as `"bonds"` in an atom structure.

```python
{"bond": "O-H", "bonds_A": ["H"], "bonds_B": []}
```

## How identifiers are built

The identifier for a structure is generated automatically.

| Dictionary | Identifier | Meaning |
|---|---|---|
| `{"atom": "O"}` | `O` | any oxygen |
| `{"atom": "O", "bonds": []}` | `O()` | oxygen with no bonds |
| `{"atom": "O", "bonds": ["H", "H"]}` | `O(H+H)` | oxygen bonded to two hydrogens (water) |
| `{"atom": "O", "bonds": ["H", "X"]}` | `O(H+X)` | oxygen bonded to one hydrogen and one additional atom of any type |
| `{"atom": "Si", "bonds": ["O", "O", {"atom": "O", "bonds": ["H"]}, {"atom": "O", "bonds": ["H"]}]}` | `Si(O+O+O(H)+O(H))` | geminal silanol |
| `{"bond": "O-H"}` | `O-H` | any O-H bond |
| `{"bond": "O-H", "bonds_A": ["H"], "bonds_B": []}` | `(H)O-H()` | O-H bond where the oxygen has one other hydrogen neighbour and the hydrogen has no other neighbours |

```{image} ../images/bond_env_plot_light.png
:class: only-light
:alt: Bonding environment plot

Bonding environments with their corresponding identifiers. In white the central atom/bond is shown and dotted lines indicate that bonds are possible but not defined by the bonding environment.
```

```{image} ../images/bond_env_plot_dark.png
:class: only-dark
:alt: Bonding environment plot

Bonding environments with their corresponding identifiers. In white the central atom/bond is shown and dotted lines indicate that bonds are possible but not defined by the bonding environment.
```


## Wildcards and performance

Deeper nested structures require a more complex search, and can increase the time it takes to search for the structure. The wildcard `"X"` expands, at match time, to every possible atom type in `atom_lib`. This can increase the number of combinations to check, especially for multiple wildcards as direct neighbours.
