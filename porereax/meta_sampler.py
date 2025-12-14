"""
Module providing parent Sampler classes

This module defines base Sampler classes, including BondSampler and AtomSampler,
which can be extended for specific sampling tasks.
"""


import numpy as np
import os
import itertools
import porereax.utils as utils


class Sampler:
    """
    Base class for samplers.
    """
    def __init__(self, name_out, dimension, process_id, atom_lib, masses, num_frames, box, **parameters):
        """
        Base sampler class.

        Parameters
        ----------
        name_out : str
            Name of the output directory of the sampler data
        dimension : str
            Dimension along which to sample.
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
        **parameters : dict
            Additional parameters for the sampler.
        """
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
        self.molecules = {}
        self.data = {}
        self.input = {}
        self.input.update({"name_out": name_out, "dimension": dimension, "box": box, "sampler_type": self.__class__.__name__})
        self.input.update(parameters)

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
        pass

    def join_samplers(self, num_cores):
        """
        Join sampler data from multiple processes.
        Parameters
        ----------
        num_cores : int
            Number of parallel processes used for sampling.
        """
        if self.process_id != -1:
            return
        data_list = {}
        for process_id in range(num_cores) if num_cores > 1 else [-1]:
            file_path = self.folder + f"/proc_{process_id}.pkl"
            proc_data = utils.load_object(file_path)
            for identifier, data in proc_data.items():
                if identifier == "input_params":
                    data_list[identifier] = data
                else:
                    if identifier not in data_list:
                        data_list[identifier] = {}
                    for key, value in data.items():
                        if key not in data_list[identifier]:
                            data_list[identifier][key] = []
                        data_list[identifier][key].append(value)
        return data_list
    
    def permutate_bonds(self, bonds, atom_lib):
        """
        Generate all permutations of bonded atom types, considering 'X' as wildcard.

        Parameters
        ----------
        bonds : list
            List of bonded atom type strings.
        atom_lib : dict
            Dictionary mapping atom type strings to their type IDs.

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
                raise ValueError(f"Error in {self.__class__.__name__}: Bonded atom {bonded_atom} not found in atom library.")
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
    
    def build_mol_dictionary(self, atom, bonds, atom_lib):
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

        Returns
        -------
        identifier : str
            Unique identifier for the molecule.
        mol : dict
            Molecule dictionary containing atom type ID and bonded atom type ID permutations.
        """
        if atom in atom_lib:
            atom_id = atom_lib[atom]
        else:
            raise ValueError(f"Error in {self.__class__.__name__}: Atom {atom} not found in atom library.")
        bonds.sort() if bonds != None else None
        identifier = atom + "+" + "_".join(bonds) if bonds != None else atom
        if bonds != None:
            bond_permutations = self.permutate_bonds(bonds, atom_lib)
        else:
            bond_permutations = None
        mol = {"atom": atom_id, "bonds": bond_permutations}
        return identifier, mol

    def get_mols(self):
        """
        Retrieve the defined molecules for sampling.

        Returns
        -------
        molecules : dict
            Dictionary of molecules defined for sampling.
        """
        return self.molecules


class AtomSampler(Sampler):
    """
    Base class for samplers that sample atoms with optional bonded atoms.
    """
    def __init__(self, name_out, dimension, atoms, process_id, atom_lib, masses, num_frames, box, **parameters):
        """
        Sampler for atoms with optional bonded atoms.
        
        Parameters
        ----------
        name_out : str
            Name of the output directory of the sampler data
        dimension : str
            Dimension along which to sample.
        atoms : list
            List of atoms to sample, each specified as a dictionary with keys:
            - "atom": str, the atom type
            - "bonds": list, optional, list of bonded atom types
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
        **parameters : dict
            Additional parameters for the sampler.
        """
        super().__init__(name_out, dimension, process_id, atom_lib, masses, num_frames, box, **parameters)
        if not isinstance(atoms, list) or len(atoms) == 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty list of atoms.")
        for atom_info in atoms:
            if "atom" not in atom_info or not isinstance(atom_info["atom"], str):
                raise ValueError(f"{self.__class__.__name__} requires each atom entry to have an 'atom' key with a string value.")
            if "bonds" in atom_info and not isinstance(atom_info["bonds"], list):
                raise ValueError(f"{self.__class__.__name__} requires the 'bonds' key to be a list if provided.")
            atom = atom_info["atom"]
            bonds = atom_info.get("bonds", None)
            identifier, mol = self.build_mol_dictionary(atom, bonds, atom_lib)
            self.molecules[identifier] = mol


class BondSampler(Sampler):
    """
    Sampler class for bonds.
    """
    def __init__(self, name_out, dimension, bonds, process_id, atom_lib, masses, num_frames, box, **parameters):
        """
        Sampler for bonds.

        Parameters
        ----------
        name_out : str
            Name of the output directory of the sampler data
        dimension : str
            Dimension along which to sample.
        bonds : list
            List of bonds to sample, each specified as a dictionary with keys:
            - "bond": str, the bond in format "A-B"
            - "bonds_A": list, optional, list of bonded atom types for atom A
            - "bonds_B": list, optional, list of bonded atom types for atom B
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
        **parameters : dict
            Additional parameters for the sampler.
        """
        super().__init__(name_out, dimension, process_id, atom_lib, masses, num_frames, box, **parameters)
        if not isinstance(bonds, list) or len(bonds) == 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty list of bonds.")
        self.bonds = {}
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
            bonds_A = bonds_A.copy() if bonds_A != None else None
            bonds_B = bonds_B.copy() if bonds_B != None else None
            bonds_A.sort() if bonds_A != None else None
            bonds_B.sort() if bonds_B != None else None
            bond_info_A = "(" + "_".join(bonds_A) + ")" if bonds_A != None else ""
            bond_info_B = "(" + "_".join(bonds_B) + ")" if bonds_B != None else ""
            identifier = bond_info_A + atom_A + "-" + atom_B + bond_info_B

            bonds_A.append(atom_B) if bonds_A != None else None
            bonds_B.append(atom_A) if bonds_B != None else None
            bonds_A.sort() if bonds_A != None else None
            bonds_B.sort() if bonds_B != None else None

            mol_identifier_A, mol_A = self.build_mol_dictionary(atom_A, bonds_A, atom_lib)
            mol_identifier_B, mol_B = self.build_mol_dictionary(atom_B, bonds_B, atom_lib)
            self.molecules[mol_identifier_A] = mol_A
            self.molecules[mol_identifier_B] = mol_B

            self.bonds[identifier] = {"bond": [atom_lib[atom_A], atom_lib[atom_B]], "mol_A": mol_identifier_A, "mol_B": mol_identifier_B}

    def get_bonds(self):
        """
        Retrieve the defined bonds for sampling.

        Returns
        -------
        bonds : dict
            Dictionary of bonds defined for sampling.
        """
        return self.bonds