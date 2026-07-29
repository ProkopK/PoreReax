Getting started
===============

Requirements
------------

* Python 3.10 or newer
* LAMMPS for running generated workflows
* ReaxFF parameter file for your target chemistry (or the provided default
  Si/O/H file)

Installation
------------

Direct installation from github:

.. code-block:: bash

   python -m pip install git+https://github.com/ProkopK/PoreReax.git

Core concepts
-------------

``Simulate``
   Builds simulation files from a GROMACS ``.gro`` structure.

``Sample``
   Loads trajectory/bond data and runs one or more analysis samplers.

``plot(...)``
   Visualizes sampled results.

Typical workflow
----------------

#. Create simulation files with ``Simulate`` and run LAMMPS.
#. Load generated outputs with ``Sample``.
#. Add desired samplers (charge, density, angle, bond, reaction, RDF, etc.).
#. Run sampling and visualize the results.