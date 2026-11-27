============
HERMES_Δ
============

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21393947-blue
   :target: https://doi.org/10.5281/zenodo.21393947
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

NetCDF writing
--------------

Writing remains serial by default (``serial_write = True``). To enable parallel
writing, set the following options in the INI configuration:

.. code-block:: ini

   serial_write = False
   parallel_io_mode = independent

``parallel_io_mode`` accepts ``independent`` (the default, with a fixed time
dimension) or ``collective`` (with an unlimited time dimension). It applies to
parallel writing; serial writing continues to use the NES serial writer.
Parallel writing requires a NetCDF installation with parallel I/O support.
Both options can also be overridden on the command line, for example
``--serial_write False --parallel_io_mode collective``.

Execution timing
----------------

Execution timing CSVs are still generated automatically in DEBUG (``log_level = 5``).
To enable timing without the other DEBUG diagnostics, set:

.. code-block:: ini

   write_time_log = True
   # Optional: relative to output_dir, or an absolute path.
   time_log_path = logs/times_<date>_<processes>.csv

``write_time_log`` defaults to ``False``. Omitting ``time_log_path`` preserves
``logs/<output_basename>_Times_<processes>.csv`` under ``output_dir``.
Setting the path alone does not enable timing. Memory records remain DEBUG-only.
Only the MPI master writes the CSV and creates its parent directories.

Custom paths support the same date tokens as ``output_name``: ``<date>`` uses
the output frequency (YYYYMMDDHH, YYYYMMDD, YYYYMM or YYYY); explicit tokens
``<YYYYMMDDHH>``, ``<YYYYMMDD>``, ``<YYYYMM>`` and ``<YYYY>`` are also supported.
``<processes>`` expands to the MPI size padded to four digits, for example 0032.
Use a date token for multi-date simulations to avoid overwriting the same CSV.

Timing breakdown
----------------

Timing logs retain the existing section names. Values accumulate over all output
time steps on each rank. Parent timers include their children; do not sum all
rows of the CSV. Aggregate per rank before computing statistics across ranks.

* ``Write/TOTAL`` is the sum of ``prepare_output``, ``write_time_step_emissions``
  and ``close_output``. The historical ``Write/Creation`` is inside ``prepare_output``.
* ``Write/write_time_step_emissions`` includes NES ``unit_conversion``,
  ``communication``, ``netcdf_write`` and ``netcdf_write_unaccounted``. These four
  components partition NES' append time; the outer timer also includes the call overhead.
* ``Calcul/TOTAL`` includes ``prepare_inventories``, ``get_time_step_emissions``
  and other calculation/logging overhead, excluding the measured writing intervals.
* ``EmissionInventory/prepare_inventories`` covers the full preparation of all
  sectors. Sector ``Speciation``, ``Vertical``, ``UpdateSectorInfo`` and
  ``PointSourcesAllocation`` describe its phases. ``Calcul/prepare_inventories``
  wraps the same operation; these two timers must not be added together.
* ``Calcul/get_time_step_emissions`` includes each sector's ``get_4d_emissions``
  and ``Calcul/combine_sectors``. Sector ``Temporal``, ``Horizontal`` and
  ``OutputUnits`` are inside ``get_4d_emissions``.

NES diagnostics are requested only when execution timing or DEBUG is enabled.
No new configuration option is required for this breakdown.

Scalability analysis
--------------------

``hermes_delta_scalability`` follows the HERMES_GR CLI options, with Delta-specific
categories. It reads historical v1.0.1 and current CSV logs without running HERMES
or MPI. After installing the updated package, for example:

.. code-block:: bash

   hermes_delta_scalability \
     --log-pattern '/path/to/logs/<replication>/times_<processes>.csv' \
     --out-dir ./scalability --summary-style simple --env MN5

The pattern requires ``<processes>`` and optionally ``<replication>``. Repetitions
are discovered automatically, including folders numbered from zero. ``--replications N``
filters the first N integer IDs, starting at zero if zero is present, otherwise one.
Incomplete process/repetition combinations are supported; duplicate combinations fail.

* ``--summary-style auto|grouped|phase``: Initialization, Calculation, Writing and
  unmeasured Other time, plus TOTAL as an independent reference.
* ``--summary-style simple``: the historical Delta categories Initialization,
  PointSourcesAllocation, Speciation, Vertical, Temporal, Horizontal and Write,
  with residual calculation/global time and TOTAL.
* ``--summary-style detailed|detail``: initialization components, Delta calculation
  categories and available NES writing diagnostics. Parents are excluded from the stack.
* ``--summary-style raw``: all measured timers, including inclusive parents,
  plotted as individual lines rather than stacked. ``Times_Detail.csv`` always
  preserves the individual timers regardless of summary style.

``--timing-stat min|mean|max|median`` defaults to max. Categories are summed per
rank before reducing ranks; stored summary columns are recomputed from rank data,
including for old logs affected by the historical mean calculation bug. Residuals
are also computed per rank. Independent maxima of different categories can still
belong to different ranks: their stacked sum need not equal the maximum TOTAL.
The TOTAL reference always uses the measured execution total; no category is
rescaled to force agreement. Inconsistent overlapping measurements raise an error
for stacked summaries; use raw to inspect them.

``--timing-schema auto|legacy|current`` defaults to auto. These values describe
Delta's log formats rather than GR's v2/v3 formats. Historical logs retain their
available measurements; missing NES diagnostics are not reconstructed.

The remaining options match GR: ``--aggregation-method mean|median`` (mean),
``--outlier-method none|iqr|mad|zscore`` (none), ``--outlier-threshold`` (1.5),
``--min-replications-for-filtering`` (4), ``--env`` and ``--skip-plots``.
Filtering uses the selected rank statistic of HERMES/TOTAL and excludes whole
runs, so all categories use the same surviving replications. This differs from
GR's independent filtering per timer.

Outputs are ``Times_Summary.csv``, ``Times_Detail.csv``, ``Outlier_Summary.csv``,
``Speedup_Summary.csv``, ``Efficiency_Summary.csv`` and three ``HERMES_*.png``
figures. Speedup is T(1)/T(p), efficiency is speedup/p. Without a one-process
baseline, times and filtering reports are still produced, with a message that
speedup/efficiency are unavailable. Zero-time metrics are left undefined.
Plotting requires matplotlib (included in the Conda environment, or install with
``pip install '.[scalability]'``); ``--skip-plots`` only requires the data dependencies.
Use a separate output directory for each experiment/style to avoid mixing reports.

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
* `Zenodo release automation <https://github.com/BSC-ES/HERMES_Delta/blob/production/ZENODO_RELEASE.rst>`_

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

* Software DOI: https://doi.org/10.5281/zenodo.21393947
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
