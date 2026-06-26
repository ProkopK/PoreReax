import pytest

from porereax.simulate import Simulate # type: ignore
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"
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
gro_charges = {
    "OM": -0.64,
    "SI": 1.28,
    "Si": 1.28,
    "O": -0.74,
    "H": 0.42,
    "OW": -1.1128,
    "HW": 0.5564,
}
atom_masses = {"Si": 28.086, "O": 15.9994, "H": 2.016}


@pytest.fixture
def simulate_pore(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Simulate(
        gro_lib=gro_lib,
        gro_charges=gro_charges,
        atom_masses=atom_masses,
        structure_file=TEST_DATA_DIR / "test_structure.gro",
    )
