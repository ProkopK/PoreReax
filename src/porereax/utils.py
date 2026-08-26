"""
Module providing utility functions for the PoreReax package.

This module includes functions for saving and loading Python objects using
pickle, loading YAML configuration files, and common mathematical operations
such as the minimum-image convention.
"""


import pickle
import re
import yaml
import numpy as np
import os


def save_object(obj, filename):
    """
    Save a Python object to a file using pickle.

    Parameters
    ----------
    obj : any
        The Python object to be saved.
    filename : str
        The path to the file where the object will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_object(file_path):
    """
    Load a Python object from a file using pickle.

    Parameters
    ----------
    file_path : str
        The path to the file from which the object will be loaded.

    Returns
    -------
    any
        The loaded Python object.
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the YAML file.

    Returns
    -------
    dict
        The contents of the YAML file as a dictionary.
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def min_image_convention(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Apply the minimal image convention to a vector given the simulation box dimensions.

    Parameters
    ----------
    vec : np.ndarray
        The input vector (shape: (N, 3)).
    box : np.ndarray
        The simulation box dimensions (shape: (3,)).

    Returns
    -------
    np.ndarray
        The vector adjusted by the minimal image convention (shape: (N, 3)).
    """
    return vec - box * np.round(vec / box)

def min_image_midpoint(vec1: np.ndarray, vec2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Calculate the midpoint between two vectors considering the minimal image convention.

    Parameters
    ----------
    vec1 : np.ndarray
        The first vector (shape: (N, 3)).
    vec2 : np.ndarray
        The second vector (shape: (N, 3)).
    box : np.ndarray
        The simulation box dimensions (shape: (3,)).

    Returns
    -------
    np.ndarray
        The midpoint vector adjusted by the minimal image convention (shape: (N, 3)).
    """
    dist = vec2 - vec1
    dist = min_image_convention(dist, box)
    midpoint = vec1 + 0.5 * dist
    midpoint %= box
    return midpoint

def get_identifiers(link_data: str) -> list:
    """
    Retrieve the list of identifiers from a data file.

    Parameters
    ----------
    link_data : str
        Path to the data file created by a sampler instance.

    Returns
    -------
    list
        List of identifiers present in the data file.
    """
    data = load_object(link_data)
    return [identifier for identifier in data.keys() if identifier != "input_params" and identifier != "num_frames"]

def get_data(link_data: str, identifier: str) -> dict:
    """
    Retrieve the data for a specific identifier from a data file.

    Parameters
    ----------
    link_data : str
        Path to the data file created by a sampler instance.
    identifier : str
        The identifier for which to retrieve the data.

    Returns
    -------
    dict
        The data corresponding to the specified identifier.
    """
    data = load_object(link_data)
    if identifier not in data:
        raise ValueError(f"Identifier '{identifier}' not found in the data file.")
    return data[identifier]

def read_pore_yml(file_path: str) -> dict:
    """
    Read a YAML file containing pore system properties and extract relevant depending on the pore shape.

    Parameters
    ----------
    file_path : str
        Path to the YAML file containing pore system properties.
    
    Returns
    -------
    dict
        A dictionary containing the extracted pore properties.
    """
    properties = {}
    system_data = load_yaml(file_path)
    if len(system_data) > 2:
        raise NotImplementedError("Only systems with one pore are supported.")
    reservoir = system_data["system"]["reservoir"]
    properties["reservoir"] = reservoir * 10
    if system_data["shape_00"]["shape"] == "CYLINDER":
        if system_data["shape_00"]["parameter"]["central"] != [0, 0, 1]:
            raise NotImplementedError("Only CYLINDER pores with central axis along z (0,0,1) are supported.")
        pore_length = 2 * system_data["system"]["centroid"][2] * 10
        box_length = system_data["system"]["dimensions"][2] * 10
        center = np.array(system_data["shape_00"]["parameter"]["centroid"]) * 10
        center[2] = box_length / 2
        gap = (box_length - pore_length - 2 * reservoir) / 2
        pore_range = np.array([reservoir + gap, box_length - reservoir - gap])
        
        properties["type"] = "cylinder"
        properties["radius"] = system_data["shape_00"]["diameter"] / 2 * 10
        properties["length"] = pore_length
        properties["center"] = center
        properties["range"] = pore_range
    else:
        raise NotImplementedError("Currently, only CYLINDER pores are supported.")

    return properties

_PLACEHOLDER_RE = re.compile(r"^([ \t]*)%\((\w+)\)s[ \t]*$", re.MULTILINE)


class Substitution:
    """Decorator that fills %(name)s placeholders in a docstring.

    Use this to inject a shared block (e.g. a common Parameters section)
    into multiple docstrings without retyping it. Unlike plain %-formatting,
    only whole-line %(name)s placeholders are replaced, so an unrelated bare
    '%' elsewhere in the docstring is left untouched. Each line of the
    replacement text is reindented to match the placeholder's own
    indentation, so it splices cleanly into an indented class docstring.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, func):
        if func.__doc__:
            def _replace(match):
                indent, key = match.group(1), match.group(2)
                if key not in self.params:
                    return match.group(0)
                value = str(self.params[key]).strip("\n")
                lines = [indent + line if line else line for line in value.splitlines()]
                return "\n".join(lines)

            func.__doc__ = _PLACEHOLDER_RE.sub(_replace, func.__doc__)
        return func
