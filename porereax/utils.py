"""
Module providing utility classes and functions for sampling molecular data.

This module defines base Sampler classes, including BondSampler and AtomSampler,
which can be extended for specific sampling tasks. It also includes functions
for saving and loading Python objects using pickle.
"""


import numpy as np
import os
import pickle
import itertools


class Sampler:
    """
    Base class for samplers.
    """
    def __init__(self, link_out: str, dimension: str, process_id=0, **parameters):
        """
        Base class for samplers.
        
        Parameters
        ----------
        link_out : str
            Output file path for the sampler data.
        dimension : str
            Dimension along which to sample.
        process_id : int, optional
            Process ID for parallel processing (default is 0).
        **parameters : dict
            Additional parameters for the sampler.
        """
        folder = ".".join(link_out.split(".")[:-1])
        if not os.path.exists(folder) and folder != "" and process_id == 0:
            os.makedirs(folder)
        self.link_out = folder + f"/proc_{process_id}." + link_out.split(".")[-1]
        self.process_id = process_id
        self.dimension = dimension
        self.input = parameters
        self.input.update({"link_out": link_out})
        self.input.update({"dimension": dimension})
        self.data = {}

    def init_sampling(self, atom_lib, dimension_params={}):
        """
        Initialize sampling data structures.
        
        Parameters
        ----------
        atom_lib : dict
            Library of atom types.
        dimension_params : dict, optional
            Additional parameters for dimension-specific sampling.
        """
        pass

    def get_data(self):
        """
        Retrieve the input parameters and sampled data.
        
        Returns
        -------
        input : dict
            Input parameters used for the sampler.
        data : dict
            Sampled data.
        """
        return self.input, self.data

    def sample(self, **parameters):
        """
        Perform sampling for a given frame.
        
        Parameters
        ----------
        **parameters : dict
            Parameters required for sampling.
        """
        pass

    @staticmethod
    def validate_inputs(inputs: dict, atom_lib: dict, sampler_type: str):
        """
        Validate common inputs for samplers.
        
        Parameters
        ----------
        inputs : dict
            Input parameters to validate.
        atom_lib : dict
            Library of atom types.
        sampler_type : str
            Type of the sampler for error messages.

        Raises
        ------
        ValueError
            If any input parameter is invalid.
        """
        if "link_out" not in inputs or not isinstance(inputs["link_out"], str):
            raise ValueError(f"{sampler_type} requires a 'link_out' string parameter.")
        if "dimension" not in inputs or not isinstance(inputs["dimension"], str):
            raise ValueError(f"{sampler_type} requires a 'dimension' string parameter.")


class BondSampler(Sampler):
    """
    Sampler class for bonds.
    """
    def __init__(self, link_out, dimension, bonds, process_id=0, **parameters):
        """
        Sampler for bonds.

        Parameters
        ----------
        link_out : str
            Output file path for the sampler data.
        dimension : str
            Dimension along which to sample.
        bonds : list
            List of bonds to sample, each specified as a string identifier.
        process_id : int, optional
            Process ID for parallel processing (default is 0).
        **parameters : dict
            Additional parameters for the sampler.
        """
        super().__init__(link_out, dimension, process_id, **parameters)
        self.bonds = {}

#     def add_bond(self, bond, constrains={}):
#         if not isinstance(bond, str):
#             raise TypeError("Bond must be a string identifier.")
#         if not isinstance(constrains, dict):
#             raise TypeError("Additional constrains musst be a dict")
#         identifier = bond
#         print(f"Adding bond {bond} to {self.__class__.__name__}")
#         self.bonds[identifier] = {"bond": bond, "constrains": constrains}

class AtomSampler(Sampler):
    """
    Sampler class for atoms with optional bonded atoms.
    """
    def __init__(self, link_out, dimension, atoms, process_id=0, **parameters):
        """
        Sampler for atoms with optional bonded atoms.
        
        Parameters
        ----------
        link_out : str
            Output file path for the sampler data.
        dimension : str
            Dimension along which to sample.
        atoms : list
            List of atoms to sample, each specified as a dict with 'atom' and optional 'bonds'.
        process_id : int, optional
            Process ID for parallel processing (default is 0).
        **parameters : dict
            Additional parameters for the sampler.
        """
        super().__init__(link_out, dimension, process_id, **parameters)
        self.molecules = {}
        for atom_info in atoms:
            atom = atom_info["atom"]
            bonds = atom_info.get("bonds", None)
            self.__add_atom(atom, bonds)

    def __add_atom(self, atom, bonds=None):
        """
        Add an atom with optional bonded atoms to the sampler.

        Parameters
        ----------
        atom : str
            The atom identifier.
        bonds : list, optional
            List of bonded atom identifiers.
        """
        if not isinstance(atom, str):
            raise TypeError("Atom must be a string identifier.")
        if not isinstance(bonds, list) and bonds is not None:
            raise TypeError("Bonds must be a list of atoms the centering atom is bonded to.")
        # if (not isinstance(bonds, (list, dict)) and bonds is not None:
        #     raise TypeError("Bonds must be a list or dict of atoms the centering atom is bonded to.")
        bonds.sort() if bonds else None
        identifier = atom + "+" + "_".join(bonds) if bonds else atom
        if self.process_id == 0:
            print(f"Adding molecule {identifier} to {self.__class__.__name__} with atom {atom} and bonds {bonds}.")
        self.molecules[identifier] = {"atom": atom, "bonds": bonds if bonds else []}

    def init_sampling(self, atom_lib, dimension_params={}):
        """
        Initialize sampling data structures for each atom and its bonded atoms.

        Parameters
        ----------
        atom_lib : dict
            Library of atom types.
        dimension_params : dict, optional
            Additional parameters for dimension-specific sampling.

        Returns
        -------
        molecules : dict
            Processed molecules with atom types and bonded atom permutations.
        """
        for identifier, bonds_info in self.molecules.items():
            atom = bonds_info["atom"]
            if atom in atom_lib:
                atom = atom_lib[atom]
            else:
                raise ValueError(f"Error in {self.__class__.__name__}: Atom {atom} not found in atom library.")
            bonds = bonds_info["bonds"]
            bond_types = []
            for bond in bonds:
                if bond in atom_lib:
                    bond_types.append(atom_lib[bond])
                else:
                    raise ValueError(f"Error in {self.__class__.__name__}: Bonded atom {bond} not found in atom library.")
            bond_permutations = list(map(list, itertools.permutations(bond_types))) # All permutations of bond types
            bond_permutations = [list(x) for x in set(tuple(bond_perm) for bond_perm in bond_permutations)] # Remove duplicates
            self.molecules[identifier].update({"atom": atom, "bonds": bond_permutations})
        return self.molecules
    
    @staticmethod
    def validate_inputs(inputs: dict, atom_lib: dict, sampler_type: str):
        """
        Validate common inputs for AtomSampler.

        Parameters
        ----------
        inputs : dict
            Input parameters to validate.
        atom_lib : dict
            Library of atom types.
        sampler_type : str
            Type of the sampler for error messages.
        """
        Sampler.validate_inputs(inputs, atom_lib, sampler_type)
        if "atoms" not in inputs or not isinstance(inputs["atoms"], list) or len(inputs["atoms"]) == 0:
            raise ValueError(f"{sampler_type} requires a non-empty list of atoms.")
        for atom_info in inputs["atoms"]:
            if "atom" not in atom_info or not isinstance(atom_info["atom"], str):
                raise ValueError(f"{sampler_type} requires each atom entry to have an 'atom' key with a string value.")
            if atom_lib is not None and atom_info["atom"] not in atom_lib:
                raise ValueError(f"{sampler_type}: Atom '{atom_info['atom']}' not found in atom library.")
            if "bonds" in atom_info and not isinstance(atom_info["bonds"], list):
                raise ValueError(f"{sampler_type} requires the 'bonds' key to be a list if provided.")
            if atom_lib is not None and atom_info["bonds"] is not None:
                for bonded_atom in atom_info["bonds"]:
                    if bonded_atom not in atom_lib:
                        raise ValueError(f"{sampler_type}: Bonded atom '{bonded_atom}' not found in atom library.")


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

def load_object(filename):
    """
    Load a Python object from a file using pickle.
    
    Parameters
    ----------
    filename : str
        The path to the file from which the object will be loaded.
        
    Returns
    -------
    any
        The loaded Python object.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)