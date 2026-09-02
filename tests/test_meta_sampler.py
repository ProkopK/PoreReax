import warnings

import numpy as np
import pytest

from porereax.meta_sampler import _build_mol_dictionary, _validate_bonds_list
from porereax.sample import _matches_bond_spec, _neighbor_atoms_excluding
from porereax.utils import load_object

atom_lib = {"A": 1, "B": 2, "C": 3, "D": 4}


class FakeBondEnumerator:
    """Minimal stand-in for ovito.data.BondsEnumerator for unit testing.

    Wraps a static (num_bonds, 2) topology array and answers
    `bonds_of_particle` from a precomputed per-atom bond-index list, exactly
    like the interface `_matches_bond_spec`/`_neighbor_atoms_excluding` rely
    on, without requiring OVITO or a real trajectory.
    """

    def __init__(self, bond_topology):
        self._bonds_per_atom = {}
        for bond_id, (a, b) in enumerate(bond_topology):
            self._bonds_per_atom.setdefault(a, []).append(bond_id)
            self._bonds_per_atom.setdefault(b, []).append(bond_id)

    def bonds_of_particle(self, atom):
        return self._bonds_per_atom.get(atom, [])


# ---------------------------------------------------------------------------
# _build_mol_dictionary / _validate_bonds_list
# ---------------------------------------------------------------------------


def test_build_mol_dictionary_flat_unchanged():
    identifier, mol = _build_mol_dictionary("B", ["A", "A"], atom_lib, "Test", True)
    assert identifier == "B(A+A)"
    assert mol["bonds"] == [[1, 1]]
    assert mol["bonds_spec"] is None


def test_build_mol_dictionary_unconstrained():
    identifier, mol = _build_mol_dictionary("B", None, atom_lib, "Test", True)
    assert identifier == "B"
    assert mol["bonds"] is None
    assert mol["bonds_spec"] is None


def test_build_mol_dictionary_nested_identifier_and_spec():
    identifier, mol = _build_mol_dictionary(
        "A",
        [{"atom": "B", "bonds": []}, "B", {"atom": "C", "bonds": ["D"]}],
        atom_lib,
        "Test",
        True,
    )
    assert identifier == "A(B+B()+C(D))"
    assert mol["bonds_spec"] is not None
    assert [spec["type_ids"] for spec in mol["bonds_spec"]] == [
        frozenset({2}),
        frozenset({2}),
        frozenset({3}),
    ]
    # Sorted order matches identifier order: "B" (unconstrained), "B()", "C(D)"
    unconstrained_b, empty_b, nested_c = mol["bonds_spec"]
    assert unconstrained_b["mol"]["bonds"] is None
    assert empty_b["mol"]["bonds"] == [[]]
    assert nested_c["mol"]["bonds"] == [[4]]


def test_build_mol_dictionary_flat_wildcard():
    identifier, mol = _build_mol_dictionary("A", ["X", "B"], atom_lib, "Test", True)
    assert identifier == "A(B+X)"
    assert mol["bonds_spec"] is None
    for other_type in atom_lib.values():
        assert [2, other_type] in mol["bonds"] or [other_type, 2] in mol["bonds"]


def test_build_mol_dictionary_nested_wildcard_spec():
    identifier, mol = _build_mol_dictionary(
        "A", [{"atom": "X", "bonds": ["D"]}], atom_lib, "Test", True
    )
    assert identifier == "A(X(D))"
    (spec,) = mol["bonds_spec"]
    assert spec["type_ids"] == frozenset(atom_lib.values())
    assert spec["mol"]["atom"] == "X"
    assert spec["mol"]["bonds"] == [[4]]


def test_matches_bond_spec_wildcard_neighbour_with_nested_requirement():
    # A (0) is bonded to a single C (1), which is itself bonded to a D (2).
    # A wildcard neighbour ("X") should match the C regardless of its type,
    # as long as C's own further bonding environment (a single "D") matches.
    local_atom_types = np.array([1, 3, 4])  # A, C, D
    local_bond_topology = np.array([[0, 1], [1, 2]])
    local_bond_enum = FakeBondEnumerator(local_bond_topology)
    _, mol = _build_mol_dictionary(
        "A", [{"atom": "X", "bonds": ["D"]}], atom_lib, "Test", True
    )
    neighbor_atoms = _neighbor_atoms_excluding(
        0, None, local_bond_topology, local_bond_enum
    )
    assert _matches_bond_spec(
        neighbor_atoms,
        local_atom_types,
        mol["bonds_spec"],
        local_bond_topology,
        local_bond_enum,
        0,
    )


def test_validate_bonds_list_accepts_nested():
    _validate_bonds_list(
        [{"atom": "B", "bonds": []}, "B", {"atom": "C", "bonds": ["D"]}], "Test"
    )


@pytest.mark.parametrize(
    "bonds",
    [
        "not_a_list",
        [{"bonds": []}],  # missing "atom" key
        [{"atom": 5}],  # "atom" not a string
        [{"atom": "B", "bonds": "not_a_list"}],
        [123],  # neither str nor dict
    ],
)
def test_validate_bonds_list_rejects_invalid(bonds):
    with pytest.raises(ValueError):
        _validate_bonds_list(bonds, "Test")


def test_bond_sampler_nested_bonds_a_identifier(sampler, tmp_path):
    path = tmp_path.as_posix()
    sampler.add_bond_length_sampling(
        name_out=path + "/nested_bond",
        bonds=[
            {
                "bond": "Si-O",
                "bonds_A": ["O", "O", {"atom": "O", "bonds": ["H"]}],
                "bonds_B": [],
            }
        ],
        dimension="Bond Length",
        num_bins=10,
        range=(0, 1),
    )
    sampler.init_samplers(sampler.sampler_inputs, -1)
    # bonds_A identifiers are sorted without the auto-appended bond partner.
    assert "(O_O_O(H))Si-O()" in sampler.bonds
    bond_info = sampler.bonds["(O_O_O(H))Si-O()"]
    mol_a = sampler.molecules[bond_info["mol_A"]]
    assert mol_a["bonds_spec"] is not None


# ---------------------------------------------------------------------------
# _matches_bond_spec / _neighbor_atoms_excluding (sample.py)
# ---------------------------------------------------------------------------

# Topology: atom 0 is "A", bonded to two "B"s (1, 2) and one "C" (3).
# C (3) is also bonded to a "D" (4). B (1) has no other bonds; B (2) is
# also bonded to a "D" (5) (so its own environment differs from B (1)'s).
atom_types = np.array([1, 2, 2, 3, 4, 4])  # A, B, B, C, D, D
bond_topology = np.array([[0, 1], [0, 2], [0, 3], [3, 4], [2, 5]])


@pytest.fixture
def bond_enum():
    return FakeBondEnumerator(bond_topology)


def test_neighbor_atoms_excluding_removes_parent_edge(bond_enum):
    others = _neighbor_atoms_excluding(3, 0, bond_topology, bond_enum)
    assert sorted(others.tolist()) == [4]


def test_neighbor_atoms_excluding_no_exclusion(bond_enum):
    others = _neighbor_atoms_excluding(0, None, bond_topology, bond_enum)
    assert sorted(others.tolist()) == [1, 2, 3]


def test_matches_bond_spec_one_b_has_no_other_bonds(bond_enum):
    _, mol = _build_mol_dictionary(
        "A",
        [{"atom": "B", "bonds": []}, "B", {"atom": "C", "bonds": ["D"]}],
        atom_lib,
        "Test",
        True,
    )
    neighbor_atoms = np.array([1, 2, 3])
    assert _matches_bond_spec(
        neighbor_atoms, atom_types, mol["bonds_spec"], bond_topology, bond_enum, 0
    )


def test_matches_bond_spec_fails_if_no_b_qualifies(bond_enum):
    # Require *both* B's to have no other bonds -- only one (atom 1) does.
    _, mol = _build_mol_dictionary(
        "A",
        [
            {"atom": "B", "bonds": []},
            {"atom": "B", "bonds": []},
            {"atom": "C", "bonds": ["D"]},
        ],
        atom_lib,
        "Test",
        True,
    )
    neighbor_atoms = np.array([1, 2, 3])
    assert not _matches_bond_spec(
        neighbor_atoms, atom_types, mol["bonds_spec"], bond_topology, bond_enum, 0
    )


def test_matches_bond_spec_fails_wrong_neighbor_count(bond_enum):
    _, mol = _build_mol_dictionary(
        "A", [{"atom": "B", "bonds": []}, "B"], atom_lib, "Test", True
    )
    neighbor_atoms = np.array([1, 2, 3])
    assert not _matches_bond_spec(
        neighbor_atoms, atom_types, mol["bonds_spec"], bond_topology, bond_enum, 0
    )


# ---------------------------------------------------------------------------
# End-to-end: nested constraint applied through the real OVITO/trajectory
# pipeline (Sample.sample), using the "sampler" fixture from conftest.py.
# ---------------------------------------------------------------------------


def test_nested_bonds_filters_real_trajectory(sampler, tmp_path):
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    path = tmp_path.as_posix()
    flat_atoms = [{"atom": "O", "bonds": ["H", "H"]}]
    nested_atoms = [
        {"atom": "O", "bonds": [{"atom": "H", "bonds": []}, {"atom": "H", "bonds": []}]}
    ]
    sampler.add_charge_sampling(path + "/flat", atoms=flat_atoms)
    sampler.add_charge_sampling(path + "/nested", atoms=nested_atoms)
    sampler.sample(is_parallel=False)

    flat_data = load_object(path + "/flat.obj")
    nested_data = load_object(path + "/nested.obj")

    flat = flat_data["O(H+H)"]
    nested = nested_data["O(H()+H())"]

    # The nested constraint ("both H neighbours have no bonds beyond the
    # O-H bond") is strictly tighter than the flat "O bonded to two H"
    # constraint, so it must never match more atoms -- and, on this
    # trajectory, it excludes some (transient bridging/proton-transfer) atoms.
    assert 0 < nested["num_atoms"] < flat["num_atoms"]
    assert nested["num_atoms"] == 18807
    assert flat["num_atoms"] == 19218
