"""
Module for parsing a static bond topology out of GROMACS input files.

This module parses the information of `.top`/`.itp` files
static topology (`[ bonds ]` entries, plus the real O-H bonds implied by a
`[ settles ]` rigid-water block) into plain numpy arrays that
:meth:`porereax.sample.Sample.from_gromacs` can inject into every frame of an
OVITO pipeline.

`[ pairs ]`, `[ angles ]`, `[ constraints ]`, `[ exclusions ]`, and
`[ virtual_sites3 ]` sections are intentionally ignored: they describe
non-bonded exclusions, angle potentials, or numerical-stability distance
constraints between atoms that are not directly bonded (e.g. the O-O
constraint in a geminal silanol), not the bond graph itself.
"""

import os
import re

import numpy as np

_SECTION_RE = re.compile(r"^\[\s*(\S+)\s*\]$")
_INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')
_STRIP_DIGITS = str.maketrans("", "", "0123456789")


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


def _iter_gromacs_lines(path):
    with open(path) as f:
        for raw_line in f:
            line = _strip_comment(raw_line)
            if line:
                yield line


def _resolve_includes(top_file: str) -> list[str]:
    """
    Recursively resolve every `#include "file.itp"` reachable from a .top
    file, relative to the directory of the file containing the directive.

    Parameters
    ----------
    top_file : str
        Path to the GROMACS `.top` file.

    Returns
    -------
    list[str]
        Paths of every (transitively) included `.itp` file, in the order
        they are first encountered.
    """
    top_dir = os.path.dirname(os.path.abspath(top_file))
    itp_files = []
    for line in _iter_gromacs_lines(top_file):
        match = _INCLUDE_RE.match(line)
        if match:
            itp_path = os.path.join(top_dir, match.group(1))
            itp_files.append(itp_path)
            itp_files.extend(_resolve_includes(itp_path))
    return itp_files


def _parse_molecules_section(top_file: str) -> list[tuple[str, int]]:
    """
    Parse the `[ molecules ]` section of a `.top` file.

    Returns
    -------
    list[tuple[str, int]]
        (moleculetype name, instance count) pairs, in file order. This order
        fixes the global atom-index layout: GROMACS requires the `.gro`
        atoms to appear in exactly this sequence.
    """
    molecules = []
    section = None
    for line in _iter_gromacs_lines(top_file):
        header = _SECTION_RE.match(line)
        if header:
            section = header.group(1).lower()
            continue
        if section == "molecules":
            name, count = line.split()[:2]
            molecules.append((name, int(count)))
    return molecules


def _parse_moleculetypes(file_path: str) -> dict[str, dict]:
    """
    Parse every `[ moleculetype ]` block in one `.top`/`.itp` file.

    Parameters
    ----------
    file_path : str
        Path to the `.top` or `.itp` file to parse.

    Returns
    -------
    dict[str, dict]
        Maps moleculetype name -> {"atoms": [atom_name, ...], "bonds":
        [(i, j), ...]}, with 0-based local atom-index bond pairs: explicit
        `[ bonds ]` entries plus the two real O-H bonds implied by a
        `[ settles ]` rigid-water block (the third, H-H, SETTLE distance is
        not a chemical bond and is not included).
    """
    moleculetypes: dict[str, dict] = {}
    section = None
    name = None
    atoms: list[str] = []
    bonds: list[tuple[int, int]] = []

    def _flush():
        if name is not None:
            moleculetypes[name] = {"atoms": atoms[:], "bonds": bonds[:]}

    for line in _iter_gromacs_lines(file_path):
        header = _SECTION_RE.match(line)
        if header:
            section = header.group(1).lower()
            if section == "moleculetype":
                _flush()
                name, atoms, bonds = None, [], []
            continue

        tokens = line.split()
        if section == "moleculetype":
            name = tokens[0]
        elif section == "atoms":
            atoms.append(tokens[4])
        elif section == "bonds":
            atom_i, atom_j = int(tokens[0]), int(tokens[1])
            bonds.append((atom_i - 1, atom_j - 1))
        elif section == "settles":
            oxygen = int(tokens[0]) - 1
            bonds.append((oxygen, oxygen + 1))
            bonds.append((oxygen, oxygen + 2))
    _flush()
    return moleculetypes


def _read_gro_atom_names(gro_file: str) -> list[str]:
    """
    Read the (numeric-suffix-stripped) atom name of every atom in a `.gro`
    file, in file order.

    Uses the same fixed-width atom-name column (11-15) and digit-stripping
    convention as `Simulate._line_mapper`, so names line up with the
    `gro_lib` dictionaries already used to drive `Simulate`.
    """
    with open(gro_file) as f:
        lines = f.readlines()
    num_atoms = int(lines[1].strip())
    return [
        line[10:15].strip().translate(_STRIP_DIGITS)
        for line in lines[2 : 2 + num_atoms]
    ]


def parse_gromacs_topology(
    gro_file: str,
    top_file: str,
    atom_lib: dict[str, int],
    gro_lib: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a static bond topology from a GROMACS `.top`/`.itp` + `.gro` pair.

    Parameters
    ----------
    gro_file : str
        Path to the `.gro` structure file (fixes atom order/count).
    top_file : str
        Path to the GROMACS `.top` file. Its `#include`d `.itp` files are
        resolved automatically, relative to the file that includes them.
    atom_lib : dict
        Library mapping atom names (e.g. "Si", "O", "H") to numeric type
        IDs, the same dictionary passed to `Sample.from_gromacs`.
    gro_lib : dict
        Maps raw `.gro` atom names (e.g. "OM", "SI", "OW", "HW", "MW") to an
        `atom_lib` name, following the same convention already used by
        `Simulate`. An empty-string value excludes the atom from sampling
        (e.g. a TIP4P `MW` virtual site): such atoms get the reserved type
        ID 0, which is never a valid `atom_lib` value.

    Returns
    -------
    atom_types : np.ndarray, shape (num_atoms,), dtype=int
        Per-atom type ID, in `.gro` atom order.
    bond_pairs : np.ndarray, shape (num_bonds, 2), dtype=int
        Global 0-based atom-index pairs for every static bond.
    """
    moleculetypes: dict[str, dict] = {}
    for itp_file in _resolve_includes(top_file):
        moleculetypes.update(_parse_moleculetypes(itp_file))
    moleculetypes.update(_parse_moleculetypes(top_file))
    print(f"Parsed {moleculetypes} from {top_file} and its includes.")

    molecules = _parse_molecules_section(top_file)
    gro_names = _read_gro_atom_names(gro_file)

    atom_types = np.zeros(len(gro_names), dtype=int)
    bond_pairs: list[tuple[int, int]] = []
    offset = 0
    for mol_name, count in molecules:
        if mol_name not in moleculetypes:
            raise ValueError(
                f"Moleculetype '{mol_name}' listed in the [ molecules ] "
                f"section of '{top_file}' was not found in any parsed "
                "[ moleculetype ] block."
            )
        mol = moleculetypes[mol_name]
        num_atoms = len(mol["atoms"])
        for _ in range(count):
            if offset + num_atoms > len(gro_names):
                raise ValueError(
                    f"'{gro_file}' has fewer atoms than the [ molecules ] "
                    f"section of '{top_file}' declares."
                )
            for local_idx, atom_name in enumerate(mol["atoms"]):
                global_idx = offset + local_idx
                expected_name = atom_name.translate(_STRIP_DIGITS)
                actual_name = gro_names[global_idx]
                if expected_name != actual_name:
                    raise ValueError(
                        f"Atom order mismatch at global index {global_idx}: "
                        f"topology expects '{expected_name}' (from "
                        f"moleculetype '{mol_name}') but '{gro_file}' has "
                        f"'{actual_name}'. The .gro atom order must match "
                        "the [ molecules ] section of the .top file."
                    )
                if actual_name not in gro_lib:
                    raise ValueError(
                        f"Atom name '{actual_name}' (index {global_idx}) "
                        "has no entry in gro_lib."
                    )
                type_name = gro_lib[actual_name]
                atom_types[global_idx] = atom_lib[type_name] if type_name else 0
            for atom_i, atom_j in mol["bonds"]:
                bond_pairs.append((offset + atom_i, offset + atom_j))
            offset += num_atoms

    if offset != len(gro_names):
        raise ValueError(
            f"Total atom count from the [ molecules ] section of "
            f"'{top_file}' ({offset}) does not match the number of atoms "
            f"in '{gro_file}' ({len(gro_names)})."
        )

    return atom_types, np.array(bond_pairs, dtype=int).reshape(-1, 2)
