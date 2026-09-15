from pathlib import Path

import numpy as np
import pytest

from porereax.gromacs_topology import parse_gromacs_topology

GROMACS_EXAMPLE_DIR = Path(__file__).parent.parent / "GROMACS_example"

atom_lib = {"Si": 1, "O": 2, "H": 3}
gro_lib = {
    "OM": "O",
    "SI": "Si",
    "Si": "Si",
    "O": "O",
    "H": "H",
    "OW": "O",
    "HW": "H",
    "MW": "",
}


@pytest.fixture
def gromacs_static_topology():
    return parse_gromacs_topology(
        GROMACS_EXAMPLE_DIR / "run" / "run.gro",
        GROMACS_EXAMPLE_DIR / "_top" / "topol.top",
        atom_lib,
        gro_lib,
    )


def test_atom_count_and_types(gromacs_static_topology):
    atom_types, _ = gromacs_static_topology
    assert atom_types.shape == (32235,)

    # OM 4255 + SI 1865 + SLX 19 + SL 344 + SLG 28*2 + SOL 6231 = 12742
    # (residue instances); per-type atom counts derived from topol.top /
    # grid.itp / tip4p2005.itp:
    num_om, num_si, num_slx, num_sl, num_slg, num_sol = 4255, 1865, 19, 344, 28, 6231
    expected_si = num_si + num_sl + num_slg
    expected_o = num_om + num_slx + num_sl + 2 * num_slg + num_sol  # + OW
    expected_h = num_sl + 2 * num_slg + 2 * num_sol  # SL/SLG silanol H + water H
    expected_excluded = num_sol  # one MW virtual site per water

    counts = dict(zip(*np.unique(atom_types, return_counts=True), strict=True))
    assert counts[atom_lib["Si"]] == expected_si
    assert counts[atom_lib["O"]] == expected_o
    assert counts[atom_lib["H"]] == expected_h
    assert counts[0] == expected_excluded


def test_bond_count_and_indices(gromacs_static_topology):
    atom_types, bond_pairs = gromacs_static_topology

    # SL: Si-O + O-H = 2 bonds; SLG: 2x(Si-O + O-H) = 4 bonds;
    # SOL (settles): O-H1 + O-H2 = 2 bonds. OM/SI/SLX contribute none.
    assert bond_pairs.shape == (344 * 2 + 28 * 4 + 6231 * 2, 2)

    num_atoms = atom_types.shape[0]
    assert bond_pairs.min() >= 0
    assert bond_pairs.max() < num_atoms

    # Every bonded atom pair must involve at least one non-excluded (real)
    # atom type on each side, and never connect two type-0 (virtual) atoms.
    assert not np.any(
        (atom_types[bond_pairs[:, 0]] == 0) & (atom_types[bond_pairs[:, 1]] == 0)
    )


def test_first_water_bonds_are_oxygen_hydrogen(gromacs_static_topology):
    atom_types, bond_pairs = gromacs_static_topology
    # The last 6231*4 atoms are SOL (OW, HW, HW, MW per molecule), and the
    # last bonds parsed are the water O-H settle-derived bonds.
    water_bonds = bond_pairs[-6231 * 2 :]
    first_oxygen = water_bonds[0, 0]
    assert atom_types[first_oxygen] == atom_lib["O"]
    assert set(water_bonds[0]) == {water_bonds[0, 0], water_bonds[0, 1]}
    assert atom_types[water_bonds[0, 1]] == atom_lib["H"]
    assert atom_types[water_bonds[1, 1]] == atom_lib["H"]
    # Both O-H bonds of the same water molecule share the same oxygen.
    assert water_bonds[0, 0] == water_bonds[1, 0]


def test_missing_gro_lib_entry_raises(tmp_path):
    incomplete_gro_lib = {k: v for k, v in gro_lib.items() if k != "MW"}
    with pytest.raises(ValueError, match="gro_lib"):
        parse_gromacs_topology(
            GROMACS_EXAMPLE_DIR / "run" / "run.gro",
            GROMACS_EXAMPLE_DIR / "_top" / "topol.top",
            atom_lib,
            incomplete_gro_lib,
        )
