import numpy as np

from porereax.sample import Sample # type: ignore
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"

def test_sample_initialization(tmp_path):
    atom_lib = {"Si": 1, "O": 2, "H": 3}
    masses = {"Si": 28.085, "O": 15.999, "H": 1.008}
    traj_file = TEST_DATA_DIR / "test_traj.lammpstrj"
    bond_file = TEST_DATA_DIR / "test_bonds.bonds"

    sampler = Sample(
        atom_lib=atom_lib,
        masses=masses,
        trajectory_file=traj_file,
        bond_file=bond_file,
    )