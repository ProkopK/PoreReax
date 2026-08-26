"""
Module providing parent Sampler classes

The module provides :class:`Sampler`, :class:`AtomSampler`, and :class:`BondSampler` as base classes for sampling various properties of MD simulations.
"""

import abc
import numpy as np
import os
import itertools
import porereax.utils as utils
import porereax.regions as regions
from porereax.utils import Substitution


_SAMPLER_INIT_PARAMS = """
dimension : str
    Dimension along which to sample.
region : str or Callable
    Region specification for sampling.
    Can be a string defining a geometric region or a function that takes coordinates and returns a boolean mask.
process_id : int
    Process ID for parallel processing.
atom_lib : dict
    Dictionary mapping atom type strings to their type IDs.
masses : dict
    Dictionary mapping atom type strings to their masses.
num_frames : int
    Total number of frames to sample.
box : np.ndarray
    Simulation box dimensions.
system_properties : dict or None
    System properties for sampling, if applicable.
"""

_NAME_OUT_PARAM = """
name_out : str
    Name of the output file of the sampler data
"""

_ATOM_SAMPLER_INIT_PARAMS = _NAME_OUT_PARAM + """
atoms : list
    List of atoms to sample, each specified as a dictionary with keys:

    - "atom": str, the atom type
    - "bonds": list, optional, list of bonded atom types
""" + _SAMPLER_INIT_PARAMS

_BOND_SAMPLER_INIT_PARAMS = _NAME_OUT_PARAM + """
bonds : list
    List of bonds to sample, each specified as a dictionary with keys:

    - "bond": str, the bond in format "A-B"
    - "bonds_A": list, optional, list of bonded atom types for atom A
    - "bonds_B": list, optional, list of bonded atom types for atom B
""" + _SAMPLER_INIT_PARAMS


def _permutate_bonds(bonds, atom_lib, class_name):
    """
    Generate all permutations of bonded atom types, considering 'X' as wildcard.

    Parameters
    ----------
    bonds : list
        List of bonded atom type strings.
    atom_lib : dict
        Dictionary mapping atom type strings to their type IDs.
    class_name : str
        Name of the calling class for error messages.

    Returns
    -------
    bond_permutations : list
        List of lists containing all permutations of bonded atom type IDs.
    """
    bond_types = []
    for bonded_atom in bonds:
        if bonded_atom in atom_lib:
            bond_types.append(atom_lib[bonded_atom])
        # elif bonded_atom == "X":
        #     bond_types.append("X")
        else:
            raise ValueError(f"Error in {class_name}: Bonded atom {bonded_atom} not found in atom library.")
    options = [atom_lib.values() if x == "X" else [x] for x in bond_types]
    expanded = itertools.product(*options)
    bond_permutations = []
    seen_permutations = set()
    for e in expanded:
        for perm in set(itertools.permutations(e)):
            if perm not in seen_permutations:
                seen_permutations.add(perm)
                bond_permutations.append(list(perm))
    return bond_permutations

def _build_mol_dictionary(atom: str, bonds, atom_lib, class_name):
    """
    Build molecule dictionary for sampling.

    Parameters
    ----------
    atom : str
        Atom type string.
    bonds : list or None
        List of bonded atom type strings or None.
    atom_lib : dict
        Dictionary mapping atom type strings to their type IDs.
    class_name : str
        Name of the calling class for error messages.

    Returns
    -------
    identifier : str
        Unique identifier for the molecule.
    mol : dict
        Molecule dictionary containing atom type ID and bonded atom type ID permutations.
    """
    if atom in atom_lib:
        atom_id = atom_lib[atom]
    elif atom == "X":
        atom_id = "X"
    else:
        raise ValueError(f"Error in {class_name}: Atom {atom} not found in atom library.")
    bonds = sorted(bonds) if bonds is not None else None
    identifier = atom + "(" + "+".join(bonds) + ")" if bonds is not None else atom
    if bonds is not None:
        bond_permutations = _permutate_bonds(bonds, atom_lib, class_name)
    else:
        bond_permutations = None
    mol = {"atom": atom_id, "bonds": bond_permutations}
    return identifier, mol

def _validate_double_atoms(doubles, class_name, attribute_name, allow_none=False):
    """
    Validate the format of double atom pairs for sampling.

    Parameters
    ----------
    doubles : list
        List of atom pairs to validate.
    class_name : str
        Name of the calling class (for error messages).
    attribute_name : str
        Name of the attribute being validated (for error messages).
    allow_none : bool, optional
        Whether to allow None values in the pairs. Default is False.

    Raises
    ------
    ValueError
        If the pairs are not in the expected format or contain invalid atom types.
    """
    if not isinstance(doubles, list) or len(doubles) == 0:
        raise ValueError(f"{class_name} '{attribute_name}' parameter must be a non-empty list.")
    for double in doubles:
        if (not isinstance(double, (list, tuple)) or len(double) != 2):
            raise ValueError(f"{class_name} '{attribute_name}' parameter must be a list of doubles (lists or tuples of length 2).")
        atom1, atom2 = double
        if (not isinstance(atom1, dict) or not isinstance(atom2, dict)) and not allow_none:
            raise ValueError(f"{class_name} '{attribute_name}' parameter must contain dictionaries with 'atom' and optional 'bonds' keys.")
        elif allow_none and (not (atom1 is None and isinstance(atom2, dict)) and
                             not (atom2 is None and isinstance(atom1, dict)) and
                             not (isinstance(atom1, dict) and isinstance(atom2, dict))):
            raise ValueError(f"{class_name} '{attribute_name}' parameter must contain dictionaries with 'atom' and optional 'bonds' keys, while one of the double can be None.")
        if (atom1 is not None and "atom" not in atom1) or (atom2 is not None and "atom" not in atom2):
            raise ValueError(f"{class_name} '{attribute_name}' parameter dictionaries must have an 'atom' key.")


@Substitution(params=_SAMPLER_INIT_PARAMS, name_out=_NAME_OUT_PARAM)
class Sampler(abc.ABC):
    """
    Base sampler class.

    Parameters
    ----------
    %(name_out)s
    %(params)s
    """
    def __init__(self, name_out, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties):
        if not isinstance(name_out, str) or name_out == "":
            raise ValueError(f"{self.__class__.__name__} requires a valid 'name_out' string parameter.")
        if not isinstance(process_id, int):
            raise ValueError(f"{self.__class__.__name__} requires an integer 'process_id' parameter.")
        if not isinstance(atom_lib, dict):
            raise ValueError(f"{self.__class__.__name__} requires a dictionary 'atom_lib' parameter.")
        if not isinstance(masses, dict):
            raise ValueError(f"{self.__class__.__name__} requires a dictionary 'masses' parameter.")
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError(f"{self.__class__.__name__} requires a positive integer 'num_frames' parameter.")
        if not isinstance(box, np.ndarray) or box.shape != (3,):
            raise ValueError(f"{self.__class__.__name__} requires a numpy array 'box' parameter with shape (3,).")
        if system_properties is not None and not isinstance(system_properties, dict):
            raise ValueError(f"{self.__class__.__name__} requires a dictionary 'system_properties' parameter or None.")
        if isinstance(region, str):
            region_function = regions.get_region_function(region, box, system_properties)
            region_name = region
        elif callable(region):
            region_function = region
            region_name = "Custom Function"
        else:
            raise ValueError(f"{self.__class__.__name__} requires a valid 'region' parameter as a string or callable function.")

        self._validate_region_function(region_function)
        self._region = region_function
        self._name_out = name_out
        self._file_out = name_out + f"_proc_{process_id}.pkl"
        self._dimension = dimension
        self._process_id = process_id
        self._atom_lib = atom_lib
        self._masses = masses
        self._num_frames = num_frames
        self._box = box
        self._system_properties = system_properties
        self._molecules = {}
        self._data = {}
        self._input = {}
        self._input.update({"name_out": name_out, "dimension": dimension, "region": region_name, "box": box, "system_properties": system_properties, "sampler_type": self.__class__.__name__})

    def save_object(self):
        """
        Save the sampler data to a file.
        """
        self._data.update({"input_params": self._input})
        utils.save_object(self._data, self._file_out)

    @abc.abstractmethod
    def sample(self, frame_id: int, mol_index: dict, mol_bonds: dict, bond_mask: dict, frame: object, bond_enum: object, positions_transformed: np.ndarray):
        """
        Sample data for the current frame.

        Parameters
        ----------
        frame_id : int
            Frame index in perspective of the subprocess (starts from 0 for each subprocess).
        mol_index : dict
            Dictionary mapping molecule identifiers to boolean masks indicating which atoms belong to that molecule in the frame.
        mol_bonds : dict
            Dictionary mapping molecule identifiers to their bonded atom indices in the frame.
        bond_mask : dict
            Dictionary mapping bond identifiers to boolean masks indicating which bonds belong to that identifier in the frame.
        frame : OVITO frame object
            Current frame object from OVITO containing atomic data.
        bond_enum : OVITO BondsEnumerator
            OVITO BondsEnumerator object for enumerating bonds in the frame.
        positions_transformed : np.ndarray
            Transformed positions of atoms in the current frame.
        """

    @abc.abstractmethod
    def join_samplers(self, num_cores: int) -> None:
        """
        Collect and combine sampler data from multiple processes. This saves the combined data to a single '.obj' file and removes the individual process files. This process is only executed by the main process (process_id == -1).

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used for sampling.
        """

    def _collect_sampler_data(self, num_cores: int) -> dict:
        """
        Collect sampler data from multiple processes.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used for sampling.

        Returns
        -------
        data_list : dict
            Dictionary containing collected data from all processes.
        """
        if self._process_id != -1:
            return {}
        data_list = {}
        for process_id in range(num_cores) if num_cores > 1 else [-1]:
            file_path = self._name_out + f"_proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            os.remove(file_path)
            input_params = proc_data.pop("input_params", None)
            data_list["input_params"] = input_params
            for identifier, data in proc_data.items():
                if identifier not in data_list:
                    data_list[identifier] = {}
                for key, value in data.items():
                    if key not in data_list[identifier]:
                        data_list[identifier][key] = []
                    data_list[identifier][key].append(value)
        return data_list

    def get_mols(self):
        """
        Retrieve the defined molecules for sampling.

        Returns
        -------
        molecules : dict
            Dictionary of molecules defined for sampling.
        """
        return self._molecules

    def _validate_region_function(self, region_function):
        """
        Validate the region function to ensure it returns a boolean mask for given coordinates.

        Parameters
        ----------
        region_function : callable
            Function that takes coordinates and returns a boolean mask.

        Raises
        ------
        ValueError
            If the region function does not return a valid boolean mask.
        """
        test_coords = np.array([[0.0, 0.0, 0.0]])
        try:
            mask = region_function(test_coords)
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (1,):
                raise ValueError(f"Region function for {self.__class__.__name__} must return a boolean numpy array of shape (N,) for input coordinates of shape (N, 3).")
        except Exception as e:
            raise ValueError(f"Error in region function for {self.__class__.__name__}: {e}")


@Substitution(params=_ATOM_SAMPLER_INIT_PARAMS)
class AtomSampler(Sampler):
    """
    Sampler class for atom-based properties.

    Parameters
    ----------
    %(params)s
    """
    def __init__(self, name_out, atoms, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties):
        super().__init__(name_out, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        if not isinstance(atoms, list) or len(atoms) == 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty list of atoms.")
        for atom_info in atoms:
            if "atom" not in atom_info or not isinstance(atom_info["atom"], str):
                raise ValueError(f"{self.__class__.__name__} requires each atom entry to have an 'atom' key with a string value.")
            if "bonds" in atom_info and not isinstance(atom_info["bonds"], list):
                raise ValueError(f"{self.__class__.__name__} requires the 'bonds' key to be a list if provided.")
            atom = atom_info["atom"]
            bonds = atom_info.get("bonds", None)
            identifier, mol = _build_mol_dictionary(atom, bonds, atom_lib, self.__class__.__name__)
            self._molecules[identifier] = mol


@Substitution(params=_BOND_SAMPLER_INIT_PARAMS)
class BondSampler(Sampler):
    """
    Sampler class for bond-based properties.

    Parameters
    ----------
    %(params)s
    """
    def __init__(self, name_out, bonds, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties):
        super().__init__(name_out, dimension, region, process_id, atom_lib, masses, num_frames, box, system_properties)
        if not isinstance(bonds, list) or len(bonds) == 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty list of bonds.")
        self._bonds = {}
        for bond_info in bonds:
            if "bond" not in bond_info or not isinstance(bond_info["bond"], str):
                raise ValueError(f"{self.__class__.__name__} requires each bond entry to have a 'bond' key with a string value.")
            if len(bond_info["bond"].split("-")) != 2:
                raise ValueError(f"{self.__class__.__name__} requires the 'bond' key to be in the format 'A-B'.")
            if "bonds_A" in bond_info and not isinstance(bond_info["bonds_A"], list):
                raise ValueError(f"{self.__class__.__name__} requires the 'bonds_A' key to be a list if provided.")
            if "bonds_B" in bond_info and not isinstance(bond_info["bonds_B"], list):
                raise ValueError(f"{self.__class__.__name__} requires the 'bonds_B' key to be a list if provided.")

            bond = bond_info["bond"]
            atom_A, atom_B = bond.split("-")
            bonds_A = bond_info.get("bonds_A", None)
            bonds_B = bond_info.get("bonds_B", None)
            bonds_A = bonds_A.copy() if bonds_A is not None else None
            bonds_B = bonds_B.copy() if bonds_B is not None else None
            if bonds_A is not None:
                bonds_A.sort()
            if bonds_B is not None:
                bonds_B.sort()
            bond_info_A = "(" + "_".join(bonds_A) + ")" if bonds_A is not None else ""
            bond_info_B = "(" + "_".join(bonds_B) + ")" if bonds_B is not None else ""
            identifier = bond_info_A + atom_A + "-" + atom_B + bond_info_B

            if bonds_A is not None:
                bonds_A.append(atom_B)
                bonds_A.sort()
            if bonds_B is not None:
                bonds_B.append(atom_A)
                bonds_B.sort()

            mol_identifier_A, mol_A = _build_mol_dictionary(atom_A, bonds_A, atom_lib, self.__class__.__name__)
            mol_identifier_B, mol_B = _build_mol_dictionary(atom_B, bonds_B, atom_lib, self.__class__.__name__)
            self._molecules[mol_identifier_A] = mol_A
            self._molecules[mol_identifier_B] = mol_B

            self._bonds[identifier] = {"bond": [atom_lib[atom_A], atom_lib[atom_B]], "mol_A": mol_identifier_A, "mol_B": mol_identifier_B}

    def get_bonds(self) -> dict:
        """
        Retrieve the defined bonds for sampling.

        Returns
        -------
        bonds : dict
            Dictionary of bonds defined for sampling.
        """
        return self._bonds