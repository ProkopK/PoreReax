import numpy as np
import os
import pickle


class Sampler:
    def __init__(self, file_name, dimension, **parameters):
        self.file_name = file_name
        self.dimension = dimension
        self.input = parameters
        self.data = {}
        pass

    def get_data(self):
        return self.input, self.data

    def sample(self, **parameters):
        pass

class BondSampler(Sampler):
    def __init__(self, file_name, dimension, **parameters):
        super().__init__(file_name, dimension, **parameters)
        pass

class AtomSampler(Sampler):
    def __init__(self, file_name, dimension, **parameters):
        super().__init__(file_name, dimension, **parameters)
        self.molecules = {}
        pass

    def add_atom(self, atom, bonds=None):
        if not isinstance(atom, str):
            raise TypeError("Atom must be a string identifier.")
        if not isinstance(bonds, list) and bonds is not None:
            raise TypeError("Bonds must be a list of atoms the centering atom is bonded to.")
        # if (not isinstance(bonds, list) or not isinstance(bonds, dict)) and bonds is not None:
        #     raise TypeError("Bonds must be a list or dict of atoms the centering atom is bonded to.")
        bonds.sort() if bonds else None
        identifier = atom + "+" + "_".join(bonds) if bonds else atom
        print(f"Adding molecule {identifier} to {self.__class__.__name__} with atom {atom} and bonds {bonds}.")
        self.molecules[identifier] = {"atom": atom, "bonds": bonds if bonds else []}
        

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