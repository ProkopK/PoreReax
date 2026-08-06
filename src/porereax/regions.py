"""
Module for defining regions in the simulation box.
"""

import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable


entry = 3


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
    elif region == "Reservoir":
        if system_properties is None:
            raise ValueError("System properties must be provided for the 'Reservoir' region.")
        return lambda coords: reservoir_region(coords, system_properties)
    elif region == "Pore":
        if system_properties is None:
            raise ValueError("System properties must be provided for the 'Pore' region.")
        return lambda coords: pore_region(coords, system_properties)
    elif region.startswith("Wall"):
        if region == "Wall":
            layer_thickness = 5
        elif region.startswith("Wall_"):
            try:
                layer_thickness = float(region.split("_")[1])
            except ValueError:
                raise ValueError(f"Invalid layer thickness specified in region identifier: {region}")
        else:
            raise ValueError(f"Unknown region identifier: {region}")
        if system_properties is None:
            raise ValueError("System properties must be provided for the 'Wall' region.")
        return lambda coords: wall_region(coords, system_properties, layer_thickness)
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

def reservoir_region(coords: NDArray[np.float64], system_properties: dict) -> NDArray[np.bool_]:
    """
    Region function that defines a reservoir region outside the pore.

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates to check.
    system_properties : dict
        System properties containing pore range, center, and radius.

    Returns
    -------
    np.ndarray
        Boolean mask indicating which coordinates are within the reservoir region.
    """
    pore_range = system_properties["range"]
    pore_center = system_properties["center"]
    pore_radius = system_properties["radius"]
    lower_bound = pore_range[0] + entry
    upper_bound = pore_range[1] - entry
    distance_from_center = np.linalg.norm(coords[:, :2] - pore_center[:2], axis=1)

    return ((coords[:, 2] <= lower_bound) | (coords[:, 2] >= upper_bound)) & (distance_from_center >= pore_radius)

def pore_region(coords: NDArray[np.float64], system_properties: dict) -> NDArray[np.bool_]:
    """
    Region function that defines a pore region inside the pore.

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates to check.
    system_properties : dict
        System properties containing pore range, center, and radius.

    Returns
    -------
    np.ndarray
        Boolean mask indicating which coordinates are within the pore region.
    """
    pore_range = system_properties["range"]
    pore_center = system_properties["center"]
    pore_radius = system_properties["radius"]
    lower_bound = pore_range[0] + entry
    upper_bound = pore_range[1] - entry
    distance_from_center = np.linalg.norm(coords[:, :2] - pore_center[:2], axis=1)

    return (coords[:, 2] > lower_bound) & (coords[:, 2] < upper_bound) & (distance_from_center < pore_radius + entry)

def wall_region(coords: NDArray[np.float64], system_properties: dict, layer_thickness: float) -> NDArray[np.bool_]:
    """
    Region function that defines a wall region around the pore.

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates to check.
    system_properties : dict
        System properties containing pore range, center, and radius.
    layer_thickness : float
        Thickness of the wall layer.

    Returns
    -------
    np.ndarray
        Boolean mask indicating which coordinates are within the wall region.
    """
    pore_range = system_properties["range"]
    pore_center = system_properties["center"]
    pore_radius = system_properties["radius"]
    lower_bound_left = pore_range[0] - layer_thickness
    upper_bound_left = pore_range[0] + layer_thickness
    lower_bound_right = pore_range[1] - layer_thickness
    upper_bound_right = pore_range[1] + layer_thickness
    distance_from_center = np.linalg.norm(coords[:, :2] - pore_center[:2], axis=1)

    reservoir_wall_left = (coords[:, 2] >= lower_bound_left) & (coords[:, 2] <= upper_bound_left) & (distance_from_center >= pore_radius - layer_thickness)
    reservoir_wall_right = (coords[:, 2] >= lower_bound_right) & (coords[:, 2] <= upper_bound_right) & (distance_from_center >= pore_radius - layer_thickness)
    pore_wall = (coords[:, 2] >= lower_bound_left) & (coords[:, 2] <= upper_bound_right) & (distance_from_center >= pore_radius - layer_thickness) & (distance_from_center <= pore_radius + layer_thickness)

    return reservoir_wall_left | reservoir_wall_right | pore_wall
