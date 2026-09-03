import numpy as np
import pytest

from porereax.molecule_structure import _build_structure_key, _structure_key_to_string

atom_lib = {"A": 1, "B": 2, "C": 3, "D": 4}
type_to_name = {v: k for k, v in atom_lib.items()}


class FakeBondEnumerator:
    """Minimal stand-in for ovito.data.BondsEnumerator for unit testing.

    Wraps a static (num_bonds, 2) topology array and answers
    `bonds_of_particle` from a precomputed per-atom bond-index list, without
    requiring OVITO or a real trajectory.
    """

    def __init__(self, bond_topology):
        self._bonds_per_atom = {}
        for bond_id, (a, b) in enumerate(bond_topology):
            self._bonds_per_atom.setdefault(a, []).append(bond_id)
            self._bonds_per_atom.setdefault(b, []).append(bond_id)

    def bonds_of_particle(self, atom):
        return self._bonds_per_atom.get(atom, [])


# ---------------------------------------------------------------------------
# _build_structure_key / _structure_key_to_string
# ---------------------------------------------------------------------------
#
# Topology: root atom 0 (A) bonded to atoms 1 (B), 2 (B), 3 (C).
# Atom 1 (B) has no further bonds.
# Atom 2 (B) is further bonded to atom 5 (D).
# Atom 3 (C) is further bonded to atom 4 (D).
#
#     1(B)
#      |
# 4(D)-3(C)-0(A)-2(B)-5(D)

_TREE_TYPES = np.array([1, 2, 2, 3, 4, 4])
_TREE_TOPOLOGY = np.array([[0, 1], [0, 2], [0, 3], [3, 4], [2, 5]])
_TREE_BOND_ENUM = FakeBondEnumerator(_TREE_TOPOLOGY)


def test_build_structure_key_steps_1_matches_flat_tuple():
    key = _build_structure_key(0, None, 1, _TREE_TYPES, _TREE_TOPOLOGY, _TREE_BOND_ENUM)
    assert key == (2, 2, 3)


def test_structure_key_to_string_steps_1_matches_legacy_format():
    key = (2, 2, 3)
    assert _structure_key_to_string(key, type_to_name, 1) == "B+B+C"


def test_build_structure_key_steps_2_nested_tuple():
    key = _build_structure_key(0, None, 2, _TREE_TYPES, _TREE_TOPOLOGY, _TREE_BOND_ENUM)
    assert key == ((2, ()), (2, (4,)), (3, (4,)))


def test_structure_key_to_string_steps_2():
    key = ((2, ()), (2, (4,)), (3, (4,)))
    assert _structure_key_to_string(key, type_to_name, 2) == "B()+B(D)+C(D)"


def test_build_structure_key_steps_3_matches_deeper_nesting():
    key = _build_structure_key(0, None, 3, _TREE_TYPES, _TREE_TOPOLOGY, _TREE_BOND_ENUM)
    # Atom 1 is a leaf (no further bonds beyond its parent), so its subtree
    # stays empty; atoms 4 and 5 are leaves one level further out, so their
    # own subtree gains one extra, empty expansion level.
    assert key == ((2, ()), (2, ((4, ()),)), (3, ((4, ()),)))
    assert _structure_key_to_string(key, type_to_name, 3) == "B()+B(D())+C(D())"


# ---------------------------------------------------------------------------
# Ring / cycle handling: only the direct parent edge is excluded at each hop,
# so an atom further out in a ring can legitimately reappear once the
# expansion travels all the way around it.
# ---------------------------------------------------------------------------

# 3-membered ring: atom 0 (A) - atom 1 (B) - atom 2 (B) - atom 0.
_RING_TYPES = np.array([1, 2, 2])
_RING_TOPOLOGY = np.array([[0, 1], [1, 2], [2, 0]])
_RING_BOND_ENUM = FakeBondEnumerator(_RING_TOPOLOGY)


def test_build_structure_key_ring_terminates_and_reflects_topology():
    # steps=2: two steps from the root stay within the ring's immediate
    # neighbours (root's type does not yet reappear).
    key_steps_2 = _build_structure_key(
        0, None, 2, _RING_TYPES, _RING_TOPOLOGY, _RING_BOND_ENUM
    )
    assert key_steps_2 == ((2, (2,)), (2, (2,)))
    assert _structure_key_to_string(key_steps_2, type_to_name, 2) == "B(B)+B(B)"

    # steps=3: three steps is enough to travel all the way around the
    # 3-membered ring, so the root atom's own type ("A") reappears as a
    # descendant -- expected given only the direct parent edge is excluded.
    key_steps_3 = _build_structure_key(
        0, None, 3, _RING_TYPES, _RING_TOPOLOGY, _RING_BOND_ENUM
    )
    assert key_steps_3 == (
        (2, ((2, (1,)),)),
        (2, ((2, (1,)),)),
    )
    assert _structure_key_to_string(key_steps_3, type_to_name, 3) == "B(B(A))+B(B(A))"


# ---------------------------------------------------------------------------
# Validation of the `steps` parameter through Sample.add_molecule_structure_sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [0, -1, 1.5])
def test_molecule_structure_sampling_invalid_steps(sampler, tmp_path, steps):
    path = tmp_path.as_posix()
    with pytest.raises(
        ValueError,
        match="MoleculeStructureSampler requires a positive integer 'steps' parameter",
    ):
        sampler.add_molecule_structure_sampling(
            name_out=path + "/test_molecule_structure", steps=steps
        )
        sampler.init_samplers(sampler.sampler_inputs, -1)


def test_molecule_structure_sampling_default_steps_is_one(sampler, tmp_path):
    path = tmp_path.as_posix()
    sampler.add_molecule_structure_sampling(name_out=path + "/test_molecule_structure")
    sampler.init_samplers(sampler.sampler_inputs, -1)
    molecule_structure_sampler = next(
        s for s in sampler.samplers if type(s).__name__ == "MoleculeStructureSampler"
    )
    assert molecule_structure_sampler._steps == 1
    assert molecule_structure_sampler._input["steps"] == 1
