"""
Module providing parent Sampler classes

The module provides :class:`Sampler`, :class:`AtomSampler`, and
:class:`BondSampler` as base classes for sampling various properties of MD
simulations.
"""

import abc
import itertools
import os

import numpy as np

import porereax.regions as regions
import porereax.utils as utils
from porereax.utils import Substitution

_SAMPLER_INIT_PARAMS = """
dimension : str
    Dimension along which to sample.
region : str or Callable
    Region specification for sampling.
    Can be a string defining a geometric region or a function that takes
    coordinates and returns a boolean mask.
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

_ATOM_SAMPLER_INIT_PARAMS = (
    _NAME_OUT_PARAM
    + """
atoms : list
    List of atoms to sample, each specified as a dictionary with keys:

    - "atom": str, the atom type
    - "bonds": list, optional, list of bonded atom types. Each entry may
      either be a plain atom type string (no constraint on what that
      neighbour is itself bonded to) or a dictionary with the same "atom"/
      "bonds" shape, describing a required bonding environment for that
      specific neighbour one hop further out. Nesting may repeat to
      describe arbitrarily deep bonding environments. An entry's atom type
      (plain string or the "atom" key of a nested dictionary) may be "X" to
      match any atom type; a nested "X" entry may still carry its own
      "bonds" requirement, constraining that wildcard neighbour's further
      bonding environment without constraining its own type.
"""
    + _SAMPLER_INIT_PARAMS
)

_BOND_SAMPLER_INIT_PARAMS = (
    _NAME_OUT_PARAM
    + """
bonds : list
    List of bonds to sample, each specified as a dictionary with keys:

    - "bond": str, the bond in format "A-B"
    - "bonds_A": list, optional, list of bonded atom types for atom A. Each
      entry may be a plain atom type string or a nested dictionary
      describing a required bonding environment for that neighbour, see
      "bonds" in :class:`AtomSampler`, including "X" as a wildcard atom
      type.
    - "bonds_B": list, optional, list of bonded atom types for atom B, with
      the same nested-dictionary support as "bonds_A".
"""
    + _SAMPLER_INIT_PARAMS
)


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
        elif bonded_atom == "X":
            bond_types.append("X")
        else:
            raise ValueError(
                f"Error in {class_name}: Bonded atom {bonded_atom} not found "
                "in atom library."
            )
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


def _normalize_bond_entry(entry, class_name):
    """
    Normalize one entry of a "bonds"/"bonds_A"/"bonds_B" list to a
    ``{"atom": str, "bonds": list or None}`` dictionary, validating its shape.

    A plain string entry (today's flat format) becomes an unconstrained
    entry; a dictionary entry may additionally carry its own nested "bonds"
    key, describing a required bonding environment for that specific
    neighbour one hop further out.

    Parameters
    ----------
    entry : str or dict
        One entry of a bonds list.
    class_name : str
        Name of the calling class for error messages.

    Returns
    -------
    normalized : dict
        Dictionary with keys "atom" (str) and "bonds" (list or None).
    """
    if isinstance(entry, str):
        return {"atom": entry, "bonds": None}
    if isinstance(entry, dict):
        if "atom" not in entry or not isinstance(entry["atom"], str):
            raise ValueError(
                f"Error in {class_name}: each nested bond entry requires an "
                "'atom' key with a string value."
            )
        bonds = entry.get("bonds", None)
        if bonds is not None and not isinstance(bonds, list):
            raise ValueError(
                f"Error in {class_name}: the nested 'bonds' key must be a "
                "list if provided."
            )
        return {"atom": entry["atom"], "bonds": bonds}
    raise ValueError(
        f"Error in {class_name}: each bond entry must be either a string or "
        "a dictionary with an 'atom' key."
    )


def _validate_bonds_list(bonds, class_name):
    """
    Recursively validate a "bonds"/"bonds_A"/"bonds_B" list.

    Parameters
    ----------
    bonds : list or None
        List of bond entries (strings or nested dictionaries), or None.
    class_name : str
        Name of the calling class for error messages.

    Raises
    ------
    ValueError
        If the list, or any nested entry within it, has an invalid shape.
    """
    if bonds is None:
        return
    if not isinstance(bonds, list):
        raise ValueError(
            f"Error in {class_name}: 'bonds' key must be a list if provided."
        )
    for entry in bonds:
        normalized = _normalize_bond_entry(entry, class_name)
        _validate_bonds_list(normalized["bonds"], class_name)


def _build_mol_dictionary(atom: str, bonds, atom_lib, class_name, initial):
    """
    Build molecule dictionary for sampling.

    Parameters
    ----------
    atom : str
        Atom type string.
    bonds : list or None
        List of bonded atom type entries (strings or nested dictionaries
        describing a required bonding environment one hop further out), or
        None.
    atom_lib : dict
        Dictionary mapping atom type strings to their type IDs.
    class_name : str
        Name of the calling class for error messages.
    initial : bool
        Whether this is the initial call (root atom) or a recursive call for
        a bonded neighbour.

    Returns
    -------
    identifier : str
        Unique identifier for the molecule.
    mol : dict
        Molecule dictionary with keys:

        - "atom": the atom type ID (or "X" for a wildcard atom).
        - "bonds": bonded atom type ID permutations (see
          :func:`_permutate_bonds`), or None if unconstrained.
        - "bonds_spec": None unless at least one bonded neighbour carries its
          own nested bonding requirement, in which case a list (one entry
          per required neighbour, in the same order as "bonds") of
          ``{"type_ids": frozenset, "mol": mol}`` dictionaries, where "mol"
          is that neighbour's own recursively built molecule dictionary.
    """
    if atom in atom_lib:
        atom_id = atom_lib[atom]
    elif atom == "X":
        if initial:
            raise ValueError(
                f"Error in {class_name}: Root atom cannot be a wildcard 'X'."
            )
        atom_id = "X"
    else:
        raise ValueError(
            f"Error in {class_name}: Atom {atom} not found in atom library."
        )

    if bonds is None:
        return atom, {"atom": atom_id, "bonds": None, "bonds_spec": None}

    normalized = [_normalize_bond_entry(entry, class_name) for entry in bonds]
    sub_results = [
        _build_mol_dictionary(
            entry["atom"], entry["bonds"], atom_lib, class_name, False
        )
        for entry in normalized
    ]
    order = sorted(range(len(sub_results)), key=lambda i: sub_results[i][0])
    sorted_identifiers = [sub_results[i][0] for i in order]
    sorted_mols = [sub_results[i][1] for i in order]
    top_level_names = [normalized[i]["atom"] for i in order]

    identifier = atom + "(" + "+".join(sorted_identifiers) + ")"
    bond_permutations = _permutate_bonds(top_level_names, atom_lib, class_name)

    if any(sub_mol["bonds"] is not None for sub_mol in sorted_mols):
        bonds_spec = [
            {
                "type_ids": frozenset(atom_lib.values())
                if name == "X"
                else frozenset({atom_lib[name]}),
                "mol": sub_mol,
            }
            for name, sub_mol in zip(top_level_names, sorted_mols, strict=True)
        ]
    else:
        bonds_spec = None

    mol = {"atom": atom_id, "bonds": bond_permutations, "bonds_spec": bonds_spec}
    return identifier, mol


def _sorted_bond_identifiers(bonds, atom_lib, class_name):
    """
    Compute the sorted list of molecule identifiers for a "bonds_A"/"bonds_B"
    list, without appending the bond partner atom.

    Used by :class:`BondSampler` to build its bond-level identifier (which
    only reflects the additional neighbours the user specified, not the bond
    partner that gets appended internally before building each endpoint's
    full molecule dictionary).

    Parameters
    ----------
    bonds : list
        List of bond entries (strings or nested dictionaries).
    atom_lib : dict
        Dictionary mapping atom type strings to their type IDs.
    class_name : str
        Name of the calling class for error messages.

    Returns
    -------
    identifiers : list
        Sorted list of molecule identifier strings, one per entry in `bonds`.
    """
    normalized = [_normalize_bond_entry(entry, class_name) for entry in bonds]
    identifiers = [
        _build_mol_dictionary(
            entry["atom"], entry["bonds"], atom_lib, class_name, False
        )[0]
        for entry in normalized
    ]
    return sorted(identifiers)


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
        If the pairs are not in the expected format or contain invalid atom
        types.
    """
    if not isinstance(doubles, list) or len(doubles) == 0:
        raise ValueError(
            f"{class_name} '{attribute_name}' parameter must be a non-empty list."
        )
    for double in doubles:
        if not isinstance(double, (list, tuple)) or len(double) != 2:
            raise ValueError(
                f"{class_name} '{attribute_name}' parameter must be a list "
                "of doubles (lists or tuples of length 2)."
            )
        atom1, atom2 = double
        if (
            not isinstance(atom1, dict) or not isinstance(atom2, dict)
        ) and not allow_none:
            raise ValueError(
                f"{class_name} '{attribute_name}' parameter must contain "
                "dictionaries with 'atom' and optional 'bonds' keys."
            )
        elif allow_none and (
            not (atom1 is None and isinstance(atom2, dict))
            and not (atom2 is None and isinstance(atom1, dict))
            and not (isinstance(atom1, dict) and isinstance(atom2, dict))
        ):
            raise ValueError(
                f"{class_name} '{attribute_name}' parameter must contain "
                "dictionaries with 'atom' and optional 'bonds' keys, while "
                "one of the double can be None."
            )
        if (atom1 is not None and "atom" not in atom1) or (
            atom2 is not None and "atom" not in atom2
        ):
            raise ValueError(
                f"{class_name} '{attribute_name}' parameter dictionaries "
                "must have an 'atom' key."
            )


@Substitution(params=_SAMPLER_INIT_PARAMS, name_out=_NAME_OUT_PARAM)
class Sampler(abc.ABC):
    """
    Base sampler class.

    Parameters
    ----------
    %(name_out)s
    %(params)s
    """

    def __init__(
        self,
        name_out,
        dimension,
        region,
        process_id,
        atom_lib,
        masses,
        num_frames,
        box,
        system_properties,
    ):
        if not isinstance(name_out, str) or name_out == "":
            raise ValueError(
                f"{self.__class__.__name__} requires a valid 'name_out' "
                "string parameter."
            )
        if not isinstance(process_id, int):
            raise ValueError(
                f"{self.__class__.__name__} requires an integer 'process_id' parameter."
            )
        if not isinstance(atom_lib, dict):
            raise ValueError(
                f"{self.__class__.__name__} requires a dictionary 'atom_lib' parameter."
            )
        if not isinstance(masses, dict):
            raise ValueError(
                f"{self.__class__.__name__} requires a dictionary 'masses' parameter."
            )
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError(
                f"{self.__class__.__name__} requires a positive integer "
                "'num_frames' parameter."
            )
        if not isinstance(box, np.ndarray) or box.shape != (3,):
            raise ValueError(
                f"{self.__class__.__name__} requires a numpy array 'box' "
                "parameter with shape (3,)."
            )
        if system_properties is not None and not isinstance(system_properties, dict):
            raise ValueError(
                f"{self.__class__.__name__} requires a dictionary "
                "'system_properties' parameter or None."
            )
        if isinstance(region, str):
            region_function = regions.get_region_function(
                region, box, system_properties
            )
            region_name = region
        elif callable(region):
            region_function = region
            region_name = "Custom Function"
        else:
            raise ValueError(
                f"{self.__class__.__name__} requires a valid 'region' "
                "parameter as a string or callable function."
            )

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
        self._input.update(
            {
                "name_out": name_out,
                "dimension": dimension,
                "region": region_name,
                "box": box,
                "system_properties": system_properties,
                "sampler_type": self.__class__.__name__,
            }
        )

    def save_object(self):
        """
        Save the sampler data to a file.
        """
        self._data.update({"input_params": self._input})
        utils.save_object(self._data, self._file_out)

    @abc.abstractmethod
    def sample(
        self,
        frame_id: int,
        molecule_mask: dict,
        molecule_bond_atoms: dict,
        bond_mask: dict,
        frame: object,
        bond_enum: object,
        positions_transformed: np.ndarray,
    ):
        """
        Sample data for the current frame.

        Parameters
        ----------
        frame_id : int
            Frame index in perspective of the subprocess (starts from 0 for
            each subprocess).
        molecule_mask : dict
            Dictionary mapping molecule identifiers to boolean masks
            indicating which atoms belong to that molecule in the frame.
        molecule_bond_atoms : dict
            Dictionary mapping molecule identifiers to their bonded atom
            indices in the frame.
        bond_mask : dict
            Dictionary mapping bond identifiers to boolean masks indicating
            which bonds belong to that identifier in the frame.
        frame : OVITO frame object
            Current frame object from OVITO containing atomic data.
        bond_enum : OVITO BondsEnumerator
            OVITO BondsEnumerator object for enumerating bonds in the frame.
        positions_transformed : np.ndarray
            Transformed positions of atoms in the current frame.
        """

    def join_samplers(self, num_cores: int) -> None:
        """
        Collect and combine sampler data from multiple processes. This saves the
        combined data to a single '<name_out>.obj' file and removes the
        individual process files. This process is only executed by the
        main process (process_id == -1).

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used for sampling.
        """
        data_list = self._collect_sampler_data(num_cores)

        combined_data = {"input_params": data_list.pop("input_params", None)}
        for identifier, data in data_list.items():
            combined_data[identifier] = self._combine_identifier(identifier, data)
        utils.save_object(combined_data, self._name_out + ".obj")

    def _iter_process_data(self, num_cores: int):
        """
        Load and yield each process's raw sampler data, removing its file afterwards.

        Only the main process (process_id == -1) yields anything; any other
        process yields nothing.

        Parameters
        ----------
        num_cores : int
            Number of parallel processes used for sampling.

        Yields
        ------
        proc_data : dict
            Raw data saved by :meth:`save_object` for one process.
        """
        if self._process_id != -1:
            return
        for process_id in range(num_cores) if num_cores > 1 else [-1]:
            file_path = self._name_out + f"_proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            os.remove(file_path)
            yield proc_data

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
        data_list = {}
        for proc_data in self._iter_process_data(num_cores):
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

    @abc.abstractmethod
    def _combine_identifier(self, identifier: str, data: dict) -> dict:
        """
        Combine data for a specific identifier across multiple processes.

        Parameters
        ----------
        identifier : str
            The identifier for which to combine data.
        data : dict
            Dictionary containing lists of data from each process for the
            given identifier.

        Returns
        -------
        combined_data : dict
            Dictionary containing the combined data for the given identifier.
        """

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
        Validate the region function to ensure it returns a boolean mask for
        given coordinates.

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
            if (
                not isinstance(mask, np.ndarray)
                or mask.dtype != bool
                or mask.shape != (1,)
            ):
                raise ValueError(
                    f"Region function for {self.__class__.__name__} must "
                    "return a boolean numpy array of shape (N,) for input "
                    "coordinates of shape (N, 3)."
                )
        except Exception as e:
            raise ValueError(
                f"Error in region function for {self.__class__.__name__}: {e}"
            ) from e


@Substitution(params=_ATOM_SAMPLER_INIT_PARAMS)
class AtomSampler(Sampler):
    """
    Sampler class for atom-based properties.

    Parameters
    ----------
    %(params)s
    """

    def __init__(
        self,
        name_out,
        atoms,
        dimension,
        region,
        process_id,
        atom_lib,
        masses,
        num_frames,
        box,
        system_properties,
    ):
        super().__init__(
            name_out,
            dimension,
            region,
            process_id,
            atom_lib,
            masses,
            num_frames,
            box,
            system_properties,
        )
        if not isinstance(atoms, list) or len(atoms) == 0:
            raise ValueError(
                f"{self.__class__.__name__} requires a non-empty list of atoms."
            )
        for atom_info in atoms:
            if "atom" not in atom_info or not isinstance(atom_info["atom"], str):
                raise ValueError(
                    f"{self.__class__.__name__} requires each atom entry to "
                    "have an 'atom' key with a string value."
                )
            _validate_bonds_list(atom_info.get("bonds"), self.__class__.__name__)
            atom = atom_info["atom"]
            bonds = atom_info.get("bonds", None)
            identifier, mol = _build_mol_dictionary(
                atom, bonds, atom_lib, self.__class__.__name__, True
            )
            self._molecules[identifier] = mol


@Substitution(params=_BOND_SAMPLER_INIT_PARAMS)
class BondSampler(Sampler):
    """
    Sampler class for bond-based properties.

    Parameters
    ----------
    %(params)s
    """

    def __init__(
        self,
        name_out,
        bonds,
        dimension,
        region,
        process_id,
        atom_lib,
        masses,
        num_frames,
        box,
        system_properties,
    ):
        super().__init__(
            name_out,
            dimension,
            region,
            process_id,
            atom_lib,
            masses,
            num_frames,
            box,
            system_properties,
        )
        if not isinstance(bonds, list) or len(bonds) == 0:
            raise ValueError(
                f"{self.__class__.__name__} requires a non-empty list of bonds."
            )
        self._bonds = {}
        for bond_info in bonds:
            if "bond" not in bond_info or not isinstance(bond_info["bond"], str):
                raise ValueError(
                    f"{self.__class__.__name__} requires each bond entry to "
                    "have a 'bond' key with a string value."
                )
            if len(bond_info["bond"].split("-")) != 2:
                raise ValueError(
                    f"{self.__class__.__name__} requires the 'bond' key to "
                    "be in the format 'A-B'."
                )
            _validate_bonds_list(bond_info.get("bonds_A"), self.__class__.__name__)
            _validate_bonds_list(bond_info.get("bonds_B"), self.__class__.__name__)

            bond = bond_info["bond"]
            atom_A, atom_B = bond.split("-")
            bonds_A = bond_info.get("bonds_A", None)
            bonds_B = bond_info.get("bonds_B", None)
            bonds_A = bonds_A.copy() if bonds_A is not None else None
            bonds_B = bonds_B.copy() if bonds_B is not None else None
            bond_info_A = (
                "("
                + "_".join(
                    _sorted_bond_identifiers(bonds_A, atom_lib, self.__class__.__name__)
                )
                + ")"
                if bonds_A is not None
                else ""
            )
            bond_info_B = (
                "("
                + "_".join(
                    _sorted_bond_identifiers(bonds_B, atom_lib, self.__class__.__name__)
                )
                + ")"
                if bonds_B is not None
                else ""
            )
            identifier = bond_info_A + atom_A + "-" + atom_B + bond_info_B

            if bonds_A is not None:
                bonds_A.append(atom_B)
            if bonds_B is not None:
                bonds_B.append(atom_A)

            mol_identifier_A, mol_A = _build_mol_dictionary(
                atom_A, bonds_A, atom_lib, self.__class__.__name__, True
            )
            mol_identifier_B, mol_B = _build_mol_dictionary(
                atom_B, bonds_B, atom_lib, self.__class__.__name__, True
            )
            self._molecules[mol_identifier_A] = mol_A
            self._molecules[mol_identifier_B] = mol_B

            self._bonds[identifier] = {
                "bond": [atom_lib[atom_A], atom_lib[atom_B]],
                "mol_A": mol_identifier_A,
                "mol_B": mol_identifier_B,
            }

    def get_bonds(self) -> dict:
        """
        Retrieve the defined bonds for sampling.

        Returns
        -------
        bonds : dict
            Dictionary of bonds defined for sampling.
        """
        return self._bonds
