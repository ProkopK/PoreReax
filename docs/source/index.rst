PoreReax
=========

PoreReax is a Python package for setting up and analysing reactive simulations with ReaxFF and LAMMPS. It can turn equilibrated ``.gro`` files into runnable simulation workflows and proviedes tools for sampling various molecular properties, and visualizing the results.

Features
--------

* **Create Simulation Workflow**: Generate molecular structures based on input GRO files. This is done using the ``Simulation`` class, which reads the ``.gro`` file and initializes the simulation environment and a ``ana.py`` file for analysis.
* **Analyse Simulation**: Analyse the results of ReaxFF simulations. The ``Sample`` class allows users to load simulation data and perform various analyses.
* **Visualize the Analyse**: The ``porereax.plot`` module offers functions to visualize the results of the analyses.


Quickstart
----------

Install the package via pip:

.. code-block:: bash

   pip install # TODO

To setup a simulation workflow from an existing ``.gro`` file run: 

.. code-block:: python

   from porereax import Simulate

   gro_lib = {"OM": "O", "OW": "O", "Si": "Si", "HW": "H", "MW": ""}
   gro_charges = {"OM": -0.64, "OW": -1.1128, "Si": 1.28, "HW": 0.5564}
   atom_masses = {"Si": 28.086, "O": 15.9994, "H": 2.016}

   sim = Simulate(gro_lib, gro_charges, atom_masses, structure_file="system.gro")
   sim.set_force_field("reax.ffield")
   sim.add_sim(type="nvt", nsteps=10000, temp=300)
   sim.generate()

The analysis script ``ana.py`` will be generated in the current working directory. To analyse the results of a simulation run, insert atoms, bonds and pairs of interest and run the script. A minimal example is shown below:

.. code-block:: python

   from porereax import Sample

   atom_lib = {'H': 1, 'O': 2, 'Si': 3}
   atom_masses = {"Si": 28.086, "O": 15.9994, "H": 2.016}

   sampler = Sample(atom_lib, atom_masses, "run_0.lammpstrj", "run_0.bonds")

   sampler.add_molecule_structure_sampling("molecule_structure")
   sampler.add_charge_sampling("charge", [{"atom": "O", bonds: ["H", "H"]}, ])
   sampler.add_bond_density_sampling("bond_density", [{"bond": "Si-O", "bonds_B": ["H"]}, ], "Cartesian1D")

   sampler.sample()

