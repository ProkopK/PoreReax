import pytest
from porereax.plot import plot_hist, plot_time, plot_2d_hist, plot_mol_structure
from porereax.utils import get_identifiers
from pathlib import Path
import matplotlib.pyplot as plt

TEST_DATA_DIR = Path(__file__).parent / "data"


def test_plot_time():
    fig, ax = plt.subplots()
    plot_time(
        TEST_DATA_DIR / "density_sampling_time.obj",
        axis=ax,
        identifiers=["O(H+Si)"],
        colors=["blue"],
        dt=100,
    )
    plot_time(TEST_DATA_DIR / "density_sampling_time.obj")

    # Test not implemented sampler and identifier
    plot_time(
        TEST_DATA_DIR / "density_sampling_time.obj",
        identifiers=["not_implemented"],
    )
    plot_time(TEST_DATA_DIR / "molecule_structures.obj")


def test_plot_mol_structure():
    plot_mol_structure(TEST_DATA_DIR / "molecule_structures.obj", identifier="O")
    with pytest.raises(ValueError):
        plot_mol_structure(TEST_DATA_DIR / "molecule_structures.obj", identifier="not_implemented")


def test_plot_2d_hist():
    plot_2d_hist(TEST_DATA_DIR / "density_sampling_2d.obj", identifier="O(H+Si)")
    plot_2d_hist(
        TEST_DATA_DIR / "bond_density_sampling_2d.obj",
        identifier="(O_O_O)Si-O(H)",
        transpose=True,
    )

    # Test not implemented sampler and identifier
    plot_2d_hist(
        TEST_DATA_DIR / "density_sampling_2d.obj",
        identifier="not_implemented",
    )
    plot_2d_hist(TEST_DATA_DIR / "molecule_structures.obj", identifier="O")


def test_plot_hist():
    # Test kwargs
    file = TEST_DATA_DIR / "angle_all_sampling.obj"
    identifiers = get_identifiers(file)
    plot_hist(
        file,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )
    plot_hist(
        file,
        colors=["blue", "orange"],
        identifiers=identifiers,
        plot_kwargs={"color": "green", "label": "test"},
    )

    # Test not implemented sampler and identifier
    plot_hist(
        file,
        identifiers=["not_implemented"],
    )
    plot_hist(
        TEST_DATA_DIR / "density_sampling_2d.obj",
    )
    plot_hist(
        TEST_DATA_DIR / "molecule_structures.obj",
    )
    plot_hist(
        file,
        identifiers=None,
        colors=None,
    )

    # Test angle plotting
    fig, ax = plt.subplots()
    file = TEST_DATA_DIR / "angle_all_sampling.obj"
    plot_hist(file)
    plot_hist(
        file,
        axis=ax,
        identifiers=["O(H+Si)", "O(H+H)"],
        colors=["blue", "orange"],
        std=True,
        mean=True,
        density=True,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )

    # Test angle plotting
    fig, ax = plt.subplots()
    file = TEST_DATA_DIR / "charge_sampling.obj"
    plot_hist(file)
    plot_hist(
        file,
        axis=ax,
        identifiers=["O(H+Si)", "O(H+H)"],
        colors=["blue", "orange"],
        std=True,
        mean=True,
        density=True,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )

    # Test bond length plotting
    fig, ax = plt.subplots()
    file = TEST_DATA_DIR / "bond_length_sampling.obj"
    plot_hist(file)
    plot_hist(
        file,
        axis=ax,
        identifiers=["(O_O_O)Si-O(H)"],
        colors=["blue", "orange"],
        std=True,
        mean=True,
        density=True,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )
    fig, ax = plt.subplots()
    file = TEST_DATA_DIR / "bond_order_sampling.obj"
    plot_hist(file)
    plot_hist(
        file,
        axis=ax,
        identifiers=["(O_O_O)Si-O(H)"],
        colors=["blue", "orange"],
        std=True,
        mean=True,
        density=True,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )

    # Test density plotting
    fig, ax = plt.subplots()
    file = TEST_DATA_DIR / "density_sampling_1d.obj"
    plot_hist(file)
    plot_hist(
        file,
        axis=ax,
        identifiers=["O(H+Si)", "O(H+H)"],
        colors=["blue", "orange"],
        std=True,
        mean=True,
        density=True,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )

    # Test rdf plotting
    fig, ax = plt.subplots()
    file = TEST_DATA_DIR / "rdf_sampling.obj"
    plot_hist(file)
    plot_hist(
        file,
        axis=ax,
        identifiers=["O(H+Si)-O(H+H)"],
        colors=["blue", "orange"],
        std=True,
        mean=True,
        density=True,
        plot_kwargs={"linestyle": "--", "linewidth": 2},
    )
