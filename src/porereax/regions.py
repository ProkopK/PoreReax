"""
Module for defining regions in the simulation box.
"""

import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


def get_region_function(region: str, box: NDArray[np.float64], system_properties: dict | None = None) -> Callable[[NDArray[np.float64]], NDArray[np.bool_]]:
    """
    Retrieve the region function based on the provided region identifier.

    Parameters
    ----------
    region : str
        The region identifier defining the region.
    box : np.ndarray
        Simulation box dimensions.
    system_properties : dict or None
        System properties for defining regions, if applicable.

    Returns
    -------
    callable
        A function that takes coordinates and returns a boolean mask.
    """
    if region == "Box":
        return box_region
    else:
        raise ValueError(f"Unknown region identifier: {region}")

def box_region(coords: NDArray[np.float64]) -> NDArray[np.bool_]:
    """
    Region function that includes all coordinates within the simulation box.

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates to check.

    Returns
    -------
    np.ndarray
        Boolean mask indicating which coordinates are inside the box.
    """
    return np.ones(coords.shape[0], dtype=bool)
