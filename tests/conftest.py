import pytest

from porereax.simulate import Simulate
from porereax.sample import Sample
from porereax.utils import load_object
from pathlib import Path
from numpy.testing import assert_array_almost_equal

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


@pytest.fixture
def sampler():
    atom_lib = {"Si": 1, "O": 2, "H": 3}
    masses = {"Si": 28.085, "O": 15.999, "H": 1.008}
    traj_file = TEST_DATA_DIR / "test_traj.lammpstrj"
    bond_file = TEST_DATA_DIR / "test_bonds.bonds"

    return Sample(
        atom_lib=atom_lib,
        masses=masses,
        trajectory_file=traj_file,
        bond_file=bond_file,
        start_end_nthframe=(0, 2, 1),
    )


def assert_input_params(obj_1, obj_2):
    assert obj_1["input_params"].keys() == obj_2["input_params"].keys(), (
        "Keys of 'input_params' do not match."
    )
    for key in obj_1["input_params"].keys():
        err_msg = f"Values for key '{key}' in 'input_params' do not match. Expected: {obj_1['input_params'][key]}, Got: {obj_2['input_params'][key]}"
        if key == "box":
            assert_array_almost_equal(
                obj_1["input_params"][key],
                obj_2["input_params"][key],
                err_msg=err_msg,
            )
        elif key == "name_out":
            continue  # skip this key as it may differ depending on the user's file naming conventions, and saved files are different for test run and create test data
        else:
            assert obj_1["input_params"][key] == obj_2["input_params"][key], err_msg


def assert_sampler_data(obj_1, obj_2, check_for_std):
    ignore_keys = ["hist_std", "mean_std"] if not check_for_std else []
    for identifier in obj_1.keys():
        if identifier == "input_params":
            continue
        elif identifier == "num_frames":
            assert obj_1[identifier] == obj_2[identifier], (
                f"Values for 'num_frames' do not match."
            )
            continue
        assert obj_1[identifier].keys() == obj_2[identifier].keys(), (
            f"Sampler data keys for identifier '{identifier}' do not match."
        )
        for data_key in obj_1[identifier].keys():
            if data_key in ignore_keys:
                continue
            assert_array_almost_equal(
                obj_1[identifier][data_key],
                obj_2[identifier][data_key],
                err_msg=f"Values for data key '{data_key}' in identifier '{identifier}' do not match. Expected: {obj_1[identifier][data_key]}, Got: {obj_2[identifier][data_key]}",
            )


def assert_sample_object_files(file_1, file_2, check_for_std=True):
    obj_1 = load_object(file_1)
    obj_2 = load_object(file_2)
    assert obj_1.keys() == obj_2.keys(), f"Keys of the two objects do not match. Expected: {obj_1.keys()}, Got: {obj_2.keys()}"
    assert_input_params(obj_1, obj_2)
    assert_sampler_data(obj_1, obj_2, check_for_std)


@pytest.fixture
def list_of_sample_object_file_names():
    return [
        "molecule_structures.obj",
        "charge_sampling.obj",
        "bond_order_sampling.obj",
        "bond_length_sampling.obj",
        "angle_all_sampling.obj",
        "angle_H-O-H_sampling.obj",
        "density_sampling_time.obj",
        "density_sampling_1d.obj",
        "density_sampling_1d_cond_charge.obj",
        "density_sampling_1d_cond_angle.obj",
        "density_sampling_2d.obj",
        "bond_density_sampling_1d.obj",
        "bond_density_sampling_2d.obj",
        "rdf_sampling.obj",
    ]
