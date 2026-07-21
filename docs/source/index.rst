PoreReax
========

PoreReax is a Python package for setting up and analysing reactive molecular dynamics workflows with ReaxFF and LAMMPS.

It helps you:

* convert equilibrated ``.gro`` structures to simulation-ready LAMMPS inputs
* configure multi-stage simulation workflows through the ``Simulate`` class
* analyse generated trajectories with the ``Sample`` class and specialized samplers
* visualize sampled results using ``porereax.plot``


.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting_started
   simulation_workflow
   analysis_workflow
