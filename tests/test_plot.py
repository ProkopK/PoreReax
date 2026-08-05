import pytest
from porereax.plot import plot
from porereax.utils import get_identifiers
from pathlib import Path
import matplotlib.pyplot as plt

TEST_DATA_DIR = Path(__file__).parent / "data"


def test_plot_not_implemented():
    with pytest.raises(ValueError):
        plot(TEST_DATA_DIR / "molecule_structures.obj", identifiers=["not_implemented"])
    with pytest.raises(ValueError):
        plot(TEST_DATA_DIR / "density_sampling_2d.obj", identifiers=["not_implemented"])
    


def test_plot(list_of_sample_object_file_names):
    for file_name in list_of_sample_object_file_names:
        plot(TEST_DATA_DIR / file_name)
        fig, ax = plt.subplots()
        plot(
            TEST_DATA_DIR / file_name,
            axis=ax,
            identifiers=get_identifiers(TEST_DATA_DIR / file_name)[1:3],
            colors=["blue", "orange"],
            std=True,
            mean=True,
            density=True,
            dt=10,
            transpose=True,
        )

def test_plot_kwargs():
    file = TEST_DATA_DIR / "angle_all_sampling.obj"
    identifiers = get_identifiers(file)
    plot(
        file,
        plot_kwargs_1d={"linestyle": "--", "linewidth": 2},
    )
    plot(
        file,
        colors=["blue", "orange"],
        identifiers=identifiers,
        plot_kwargs_1d={"color": "green", "label": "test"},
    )

    plot(
        file,
        identifiers=None,
        colors=None,
    )
