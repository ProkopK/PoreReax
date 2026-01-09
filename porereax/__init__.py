"""
PoreReax is a Python package designed for analyzing and setting up reactive molecular dynamics simulations using the ReaxFF force field. It provides tools for setting up simulations, sampling various molecular properties, and visualizing the results.
"""


import porereax.simulate as simulate
import porereax.utils as utils
import porereax.plot as plot

import porereax.density as density
import porereax.charge as charge
import porereax.angle as angle
import porereax.bond_length as bond_length
import porereax.molecule_structure as molecule_structure
import porereax.rdf as rdf

from porereax.sample import Sample


__version__ = "0.0.1"

__all__ = [
    "simulate",
    "utils",
    "plot",
    "density",
    "charge",
    "angle",
    "bond_length",
    "molecule_structure",
    "rdf",
    "Sample",
    "__version__",
]
