"""
Module providing parent Sampler classes

This module defines base Sampler classes, including BondSampler and AtomSampler,
which can be extended for specific sampling tasks.
"""


import numpy as np
import os
import itertools


class Sampler:
    """
    Base class for samplers.
    """
    def __init__(self, name_out, dimension, process_id, atom_lib, masses, num_frames, box, **parameters):
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
        self.folder = name_out
        if not os.path.exists(self.folder) and self.folder != "" and process_id == -1:
            os.makedirs(self.folder)
        self.file_out = self.folder + f"/proc_{process_id}.pkl"
        self.dimension = dimension
        self.process_id = process_id
        self.atom_lib = atom_lib
        self.masses = masses
        self.num_frames = num_frames
        self.box = box
        self.input = {}
        self.input.update({"name_out": name_out, "dimension": dimension})
        self.input.update(parameters)
        self.data = {}

    def sample(self, **parameters):
        pass

    def join_samplers(self, num_cores):
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


class AtomSampler(Sampler):
    """
    Sampler class for atoms with optional bonded atoms.
    """
    def __init__(self, name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, **parameters):
        """
        Sampler for atoms with optional bonded atoms.
        
        Parameters
        ----------
        name_out : str
            Name of the output directory and object file of the sampler data
        dimension : str
            Dimension along which to sample.
        atoms : list
            List of atoms to sample, each specified as a dict with 'atom' and optional 'bonds'.
        process_id : int, optional
            Process ID for parallel processing (default is 0).
        **parameters : dict
            Additional parameters for the sampler.
        """
        if not isinstance(atoms, list) or len(atoms) == 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty list of atoms.")
        super().__init__(name_out, dimension, process_id, atom_lib, masses, num_frames, box, **parameters)
        self.molecules = {}
        for atom_info in atoms:
            if "atom" not in atom_info or not isinstance(atom_info["atom"], str):
                raise ValueError(f"{self.__class__.__name__} requires each atom entry to have an 'atom' key with a string value.")
            if "bonds" in atom_info and not isinstance(atom_info["bonds"], list):
                raise ValueError(f"{self.__class__.__name__} requires the 'bonds' key to be a list if provided.")
            atom = atom_info["atom"]
            bonds = atom_info.get("bonds", None)
            bonds.sort() if bonds else None
            identifier = atom + "+" + "_".join(bonds) if bonds else atom
            if self.process_id == -1:
                print(f"Adding molecule {identifier} to {self.__class__.__name__} with atom {atom} and bonds {bonds}.")
            self.molecules[identifier] = {"atom": atom, "bonds": bonds if bonds else []}
        print()

        atoms = atom_lib.values()
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
                elif bond == "X":
                    bond_types.append("X")
                else:
                    raise ValueError(f"Error in {self.__class__.__name__}: Bonded atom {bond} not found in atom library.")
            options = [atoms if x == "X" else [x] for x in bond_types]
            expanded = itertools.product(*options)
            bond_permutations = []
            seen_permutations = set()
            for e in expanded:
                for perm in set(itertools.permutations(e)):
                    if perm not in seen_permutations:
                        seen_permutations.add(perm)
                        bond_permutations.append(list(perm))
            self.molecules[identifier].update({"atom": atom, "bonds": bond_permutations})

    def get_mols(self):
        return self.molecules


class BondSampler(Sampler):
    """
    Sampler class for bonds.
    """
    pass
#     def __init__(self, name_out, dimension, bonds, process_id=0, **parameters):
#         """
#         Sampler for bonds.

#         Parameters
#         ----------
#         name_out : str
#             Name of the output directory and object file of the sampler data
#         dimension : str
#             Dimension along which to sample.
#         bonds : list
#             List of bonds to sample, each specified as a string identifier.
#         process_id : int, optional
#             Process ID for parallel processing (default is 0).
#         **parameters : dict
#             Additional parameters for the sampler.
#         """
#         super().__init__(name_out, dimension, process_id, **parameters)
#         self.bonds = {}

#     def add_bond(self, bond, constrains={}):
#         if not isinstance(bond, str):
#             raise TypeError("Bond must be a string identifier.")
#         if not isinstance(constrains, dict):
#             raise TypeError("Additional constrains musst be a dict")
#         identifier = bond
#         print(f"Adding bond {bond} to {self.__class__.__name__}")
#         self.bonds[identifier] = {"bond": bond, "constrains": constrains}