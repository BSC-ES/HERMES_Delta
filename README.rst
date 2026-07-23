============
HERMES_Δ
============

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21393948-blue
   :target: https://doi.org/10.5281/zenodo.21393948
   :alt: DOI

.. image:: https://img.shields.io/badge/license-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0.html
   :alt: License: GPL v3

**HERMES_Δ (High-Elective Resolution Modelling Emission System – Delta)**
is an open-source, Python-based emission processing system designed to transform
officially reported emission inventories into model-ready datasets for air
quality and greenhouse gas (GHG) simulations.

It provides high-resolution **spatial**, **temporal**, **vertical**, and
**chemical speciation** processing, enabling integration with state-of-the-art
atmospheric models for research and regulatory applications.

----

Main Features
-------------

* 🔍 **Inventory processing:** Works with official emission inventories reported under the CLTRAP and UNFCCC.
* 🗺️ **High-resolution horizontal spatial disaggregation** using vector and raster proxies.
* ⏱️ **Flexible temporal allocation**, up to the hourly resolution.
* 🧪 **Vertical distribution** taking into account specific stack height information for point sources.
* 🧪 **Chemical speciation** of primary emissions following multiple chemical mechanisms.
* ⚙️ **Parallel execution** (MPI-based) for efficient HPC workflows.
* 🌍 **Output formats** compatible with state-of-the-art atmospheric chemistry transport models, including:

  * CF-COMPLIENT
  * CHIMERE
  * CMAQ
  * WRF-Chem
  * MONARCH
  * MOCAGE

* 🧰 **Configurable and reproducible workflows** via INI/YAML configuration.

----

Dependencies
------------

HERMES_Δ requires the `following software and Python packages <https://github.com/BSC-ES/HERMES_Delta/blob/production/environment.yml>`_, which can be
installed via Conda:

.. code-block:: yaml

  - python=3.10
  - libnetcdf=*=mpi_mpich*
  - netCDF4=*=mpi_mpich*
  - h5py=*=mpi_mpich*
  - pytest
  - pytest-cov
  - pycodestyle>=2.10.0
  - geopandas>=0.10.2
  - rtree>=0.9.0
  - numpy>=1.20.0
  - pyproj
  - setuptools>=66.1.1
  - pytest>=7.2.1
  - shapely
  - mpi4py ~= 3.1.4
  - eccodes
  - python-eccodes
  - filelock
  - configargparse
  - psutil
  - pyyaml
  - gdal
  - rasterio>=1.4
  - holidays
  - pip


Some packages require MPI-enabled builds (e.g., `mpi_mpich` variants).

----

Quick Start
-----------

Clone the repository and install dependencies using the provided `Makefile`:

.. code-block:: bash

   git clone https://github.com/BSC-ES/HERMES_Delta.git
   cd HERMES_Delta
   make full_installation

This installs a Conda environment with all required packages.

For advanced installation on HPC systems, see
`the installation instructions <https://github.com/BSC-ES/HERMES_Delta/wiki/HERMES_Delta_Installation_Makefile>`_.

----

Running HERMES_Δ
----------------

The system is run from the command line with a configuration file:

.. code-block:: bash

    hermes_delta --my-config /path/to/your_config.ini


* Parameters can be defined in the INI file or overridden via the command line.
* When a parameter is defined in both, the command-line value takes precedence.

Configuration details:
`HERMES_Delta_HowToConfigure <https://github.com/BSC-ES/HERMES_Delta/wiki/HERMES_Delta_HowToConfigure>`_

----

Documentation
-------------

Full documentation is available in `the project Wiki <https://github.com/BSC-ES/HERMES_Delta/wiki>`_ and companion guides:

* `📘 User Guide / Configuration <https://github.com/BSC-ES/HERMES_Delta/wiki/HERMES_Delta_HowToConfigure>`_
* `📄 Input file formats <https://github.com/BSC-ES/HERMES_Delta/wiki/HERMES_Delta_InputFiles>`_
* `🛠️ Installation guide <https://github.com/BSC-ES/HERMES_Delta/wiki/HERMES_Delta_Installation_Makefile>`_
* `📊 Short examples <https://github.com/BSC-ES/HERMES_Delta/wiki/HERMES_Delta_ShortExamples>`_
* `📝 Change log <https://github.com/BSC-ES/HERMES_Delta/blob/production/CHANGELOG.rst>`_

----

Availability and License
------------------------

HERMES_Δ is licensed under the **GNU General Public License (GPL) version 3**.
See the `LICENSE <https://github.com/BSC-ES/HERMES_Delta/blob/production/LICENSE.rst>`_ file for details.

----

How to Cite
-----------

If you use **HERMES_Δ** in your research, please cite the following software release:

   Tena Medina, C., Gehlen, J., Rizza, L., & Guevara, M. (2026).
   HERMES_Δ: High-Elective Resolution Modelling Emission System – Delta (Version v1.0.1)
   [Computer software]. Zenodo.
   https://doi.org/10.5281/zenodo.21393948

and the following scientific paper:

   Guevara, M., Tena, C., Camps, P., Oliveira, K., Gehlen, J., Albarracin, A., Castesana, P., Collado, O., Herrero, L., Legarreta, O., Lombardich, I., Piñero-Megías, C., Rizza, L., Slater, J., Viñas, A., Macchia, F., Montane, G., Jorba, O., and Pérez García-Pando, C. (2026)
   HERMES_Δ v1.0.1: an open-source emission processor for translating official air pollutant and greenhouse gas inventories into model-ready emissions, 
   EGUsphere [preprint], 
   https://doi.org/10.5194/egusphere-2026-4146, 2026.

----

Acknowledgements
----------------

This work was developed as part of the
`high-Resolution air Emissions Systems to suPport modellIng and monitoRing Efforts (RESPIRE)
national project <https://respire.bsc.es>`_.

RESPIRE is part of the **Plan de Recuperación, Transformación y Resiliencia (PRTyR)**,
funded by the **European Union – NextGenerationEU**.

The development team acknowledges support from:

.. image:: https://raw.githubusercontent.com/wiki/BSC-ES/HERMES_Delta/LogosRESPIRE.png
   :alt: Project logos
   :align: center

* `Funded by the European Commission – NextGenerationEU <https://next-generation-eu.europa.eu>`_
* `Government of Spain – Ministerio para la Transición Ecológica <https://www.miteco.gob.es>`_
* `Agencia Estatal de Meteorología (AEMET) <https://www.aemet.es>`_
* `Barcelona Supercomputing Center (BSC) <https://www.bsc.es>`_

----

Related Resources
-----------------

* Software DOI: https://doi.org/10.5281/zenodo.21393948
* Scientific paper DOI: https://doi.org/10.5194/egusphere-2026-4146
* Source code: https://github.com/BSC-ES/HERMES_Delta
* Documentation: https://github.com/BSC-ES/HERMES_Delta/wiki
* Benchmark data: https://doi.org/10.5281/zenodo.21471814

----

Contact & Support
-----------------

For questions, support or contributions:

* `Carles Tena <https://www.bsc.es/es/tena-carles>`_ – carles.tena@bsc.es
* `Marc Guevara <https://www.bsc.es/guevara-marc>`_ – marc.guevara@bsc.es
* `Johanna Gehlen <https://www.bsc.es/gehlen-johanna>`_ – johanna.gehlen@bsc.es
* `Luca Rizza <https://www.bsc.es/rizza-luca>`_ – luca.rizza@bsc.es

Contributions welcome via GitHub pull requests!
