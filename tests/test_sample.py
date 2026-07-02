import pytest
import warnings

from porereax.sample import Sample
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"
atom_lib = {"Si": 1, "O": 2, "H": 3}
masses = {"Si": 28.085, "O": 15.999, "H": 1.008}
traj_file = TEST_DATA_DIR / "test_traj.lammpstrj"
bond_file = TEST_DATA_DIR / "test_bonds.bonds"
start_end_nthframe = (0, 10, 1)

atoms = [
    {"atom": "O", "bonds": ["H", "H"]},
    {"atom": "O", "bonds": ["Si", "H"]},
    {"atom": "O", "bonds": ["H"]},
    {"atom": "Si", "bonds": ["O", "O", "O", "O"]},
    {"atom": "Si", "bonds": ["H", "H", "H", "H"]},
]
bonds = [{"bond": "Si-O", "bonds_A": ["O", "O", "O"], "bonds_B": ["H"]}]


def test_sample_initialization():
    sampler = Sample(
        atom_lib=atom_lib,
        masses=masses,
        trajectory_file=traj_file,
        bond_file=bond_file,
    )


@pytest.mark.parametrize(
    "kwargs, expected_exception, expected_message",
    [
        # ({"atom_lib": [1, 2, 3]}, TypeError, "atom_lib must be a dictionary"),
        (
            {"atom_lib": {"Si": 1, "H": 3}},
            ValueError,
            "do not match those in atom_lib",
        ),
        (
            {"atom_lib": {"Si": 1, "O": 2, "H": 3, "O2": 2}},
            ValueError,
            "atom_lib must have a one-to-one",
        ),
        (
            {"masses": {"Si": 28.085, "O": 15.999}},
            ValueError,
            "keys must match atom_lib",
        ),
        (
            {"start_end_nthframe": (4, 2, 1)},
            ValueError,
            "Invalid start_end frame range",
        ),
    ],
)
def test_sample_initialization_validation(kwargs, expected_exception, expected_message):
    with pytest.raises(expected_exception, match=expected_message):
        Sample(
            atom_lib=kwargs.get("atom_lib", atom_lib),
            masses=kwargs.get("masses", masses),
            trajectory_file=traj_file,
            bond_file=bond_file,
            start_end_nthframe=kwargs.get("start_end_nthframe", start_end_nthframe),
        )


@pytest.fixture
def full_sampler(sampler, tmp_path):
    path = tmp_path.as_posix()
    sampler.add_molecule_structure_sampling(path + "/molecule_structures")
    sampler.add_charge_sampling(
        path + "/charge_sampling",
        atoms=atoms,
    )
    sampler.add_bond_length_sampling(
        path + "/bond_length_sampling",
        bonds=bonds,
        dimension="Bond Length",
    )
    sampler.add_angle_sampling(
        path + "/angle_all_sampling", atoms=atoms, num_bins=180, angle="all"
    )
    sampler.add_angle_sampling(
        path + "/angle_H-O-H_sampling", atoms=atoms, num_bins=90, angle="H-O-H"
    )
    return sampler


def test_sample_sampling_parallel(full_sampler):
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    full_sampler.sample(is_parallel=True)


def test_sample_sampling_serial(full_sampler):
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    full_sampler.sample(is_parallel=False)


def test_sample_ovito_conflicts(tmp_path):
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    import ovito

    with pytest.raises(RuntimeError):
        sampler = Sample(
            atom_lib=atom_lib,
            masses=masses,
            trajectory_file=traj_file,
            bond_file=bond_file,
        )
