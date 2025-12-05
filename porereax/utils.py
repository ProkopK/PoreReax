import numpy as np
import os
import pickle
import itertools


class Sampler:
    def __init__(self, file_name, dimension, **parameters):
        self.file_name = file_name
        self.dimension = dimension
        self.input = parameters
        self.input.update({"file_name": file_name})
        self.input.update({"dimension": dimension})
        self.data = {}
        pass

    def init_sampling(self, atom_lib, dimension_params={}):
        pass

    def get_data(self):
        return self.input, self.data

    def sample(self, **parameters):
        pass

class BondSampler(Sampler):
    def __init__(self, file_name, dimension, **parameters):
        super().__init__(file_name, dimension, **parameters)
        self.bonds = {}

    def add_bond(self, bond, constrains={}):
        if not isinstance(bond, str):
            raise TypeError("Bond must be a string identifier.")
        if not isinstance(constrains, dict):
            raise TypeError("Additional constrains musst be a dict")
        identifier = bond
        print(f"Adding bond {bond} to {self.__class__.__name__}")
        self.bonds[identifier] = {"bond": bond, "constrains": constrains}

class AtomSampler(Sampler):
    def __init__(self, file_name, dimension, **parameters):
        super().__init__(file_name, dimension, **parameters)
        self.molecules = {}

    def add_atom(self, atom, bonds=None):
        if not isinstance(atom, str):
            raise TypeError("Atom must be a string identifier.")
        if not isinstance(bonds, list) and bonds is not None:
            raise TypeError("Bonds must be a list of atoms the centering atom is bonded to.")
        # if (not isinstance(bonds, (list, dict)) and bonds is not None:
        #     raise TypeError("Bonds must be a list or dict of atoms the centering atom is bonded to.")
        bonds.sort() if bonds else None
        identifier = atom + "+" + "_".join(bonds) if bonds else atom
        print(f"Adding molecule {identifier} to {self.__class__.__name__} with atom {atom} and bonds {bonds}.")
        self.molecules[identifier] = {"atom": atom, "bonds": bonds if bonds else []}

    def init_sampling(self, atom_lib, dimension_params={}):
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