import numpy as np


def get_region_function(region, box, system):
    """
    Retrieve the region function based on the provided region identifier.

    Parameters
    ----------
    region : str
        The region identifier defining the region.
    box : np.ndarray
        Simulation box dimensions.
    system : dict or None
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
    
def box_region(coords):
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
