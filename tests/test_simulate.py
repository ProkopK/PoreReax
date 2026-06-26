import pytest

from porereax.simulate import Simulate
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"


def test_simulate_pore_native(simulate_pore, tmp_path):
    assert isinstance(simulate_pore, Simulate)
    simulate_pore.generate()
    assert (tmp_path / "system.data").exists()
    assert (tmp_path / "reax.ffield").exists()
    assert (tmp_path / "ana.py").exists()


def test_simulate_pore_with_setup(simulate_pore, tmp_path):
    with pytest.raises(FileNotFoundError):
        simulate_pore.set_job_file(
            file_path="non_existent_file.txt", submit_command="s", lammps_command="s"
        )
    with pytest.raises(ValueError):
        simulate_pore.set_job_file(
            file_path=TEST_DATA_DIR / "test_job.job",
            submit_command="",
            lammps_command="s",
        )
    with pytest.raises(ValueError):
        simulate_pore.set_job_file(
            file_path=TEST_DATA_DIR / "test_job.job",
            submit_command="s",
            lammps_command="",
        )
    with pytest.raises(FileNotFoundError):
        simulate_pore.set_force_field("non_existent_file.txt")

    job_file_path = TEST_DATA_DIR / "test_job.job"
    submit_command = "sbatch"
    lammps_command = "mpirun --bind-to none --map-by node lmp -in {input_file} -log {log_file} -k on -sf kk -pk kokkos neigh half newton on comm host"
    simulate_pore.set_job_file(
        file_path=job_file_path,
        submit_command=submit_command,
        lammps_command=lammps_command,
    )
    simulate_pore.set_force_field(TEST_DATA_DIR / "test_reax.ffield")

    simulate_pore.add_image_dump()

    simulate_pore.add_image_dump(atom_colors={"Si": "orange", "O": "red", "H": "white"}, 
    atom_sizes={"Si": 3, "O": 2, "H": 1}, 
    kwargs="shiny 0.1 box no 0.01")
    simulate_pore.add_image_dump(map_by_charge="amap -1 2 ca 0.0 3 min royalblue 0 green max orangered")

    simulate_pore.add_sim("nvt", 20000, 300)
    simulate_pore.add_sim("nvt", 20000, 300)

    simulate_pore.generate()

    assert (tmp_path / "run_0.lmp").exists()
    assert (tmp_path / "run_0.job").exists()
    assert (tmp_path / "run_1.lmp").exists()
    assert (tmp_path / "run_1.job").exists()