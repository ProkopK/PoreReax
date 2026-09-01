"""
PoreReax is a Python package designed for analysing and setting up reactive
molecular dynamics simulations using the ReaxFF force field. It provides tools
for setting up simulations, sampling various molecular properties, and
visualizing the results.
"""

from porereax.plot import plot_data
from porereax.sample import Sample
from porereax.simulate import Simulate
from porereax.utils import get_data, get_identifiers, load_object

__version__ = "0.0.1"

__all__ = [
    "Sample",
    "Simulate",
    "__version__",
    "get_data",
    "get_identifiers",
    "load_object",
    "plot_data",
]
