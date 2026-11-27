============
CHANGELOG
============

.. start-here

1.0.11
============

* Release date: 2026/11/27
* Changes and new features:

  * Added GitLab CI automation to publish stable releases as new versions of the existing
    Zenodo Concept DOI, after tests and the GitHub mirror succeed.
  * Added release documentation for configuring credentials, validating publications and
    recovering interrupted drafts.

1.0.10
============

* Release date: 2026/10/23
* Changes and new features:

  * Updated the bundled NES source dependency to v1.3.6, including optimized serial time-step writing,
    optional writing timing diagnostics, parallel I/O modes and corrected CMAQ/WRF-Chem metadata on MPI ranks.
  * Added ``parallel_io_mode`` to select ``independent`` or ``collective`` parallel NetCDF writing.
    The default is ``independent``, preserving the behavior with NES v1.3.6 when the option is omitted.
    ``serial_write`` still defaults to ``True``; the parallel mode applies when ``serial_write=False``.
    Independent mode uses a fixed time dimension; collective mode uses an unlimited time dimension.
  * Added ``write_time_log=False`` to enable execution timing CSVs independently of DEBUG.
    DEBUG continues to record times automatically; memory diagnostics remain DEBUG-only.
  * Added optional ``time_log_path`` (absolute or relative to ``output_dir``), with date tokens
    and ``<processes>`` substitution. Omitting it preserves the historical CSV path and filename.
    Custom parent directories are created by the MPI master only when writing the log.
  * Fixed timing and memory summary statistics to use only MPI rank columns, excluding previously
    calculated summary columns from the mean and maximum.
  * Added detailed output timing from NES for unit conversion, communication, NetCDF writing and
    unaccounted time, requested only when timing or DEBUG is enabled. Added preparation, append and
    close timers while preserving the existing ``Write/TOTAL`` and ``Write/Creation`` entries.
  * Added calculation timing for inventory preparation, per-sector 4D emission generation, sector
    combination and output unit metadata. Split sector metadata updates from point-source allocation.
  * Fixed ``EmissionInventory/prepare_inventories`` to measure the complete operation across all sectors
    instead of the final sub-operation. Documented inclusive parent timers to avoid double counting.
  * Added ``hermes_delta_scalability`` with GR-compatible discovery, replication, filtering,
    aggregation and plotting options, using Delta's historical and current timing categories.
    Supports grouped, simple, detailed and raw summaries, CSV outputs, speedup and efficiency.
    Categories are combined per MPI rank before computing rank statistics; run-level outlier
    filtering keeps a common replication sample across categories. Raw overlapping timers are
    plotted as lines, and missing one-process baselines skip speedup/efficiency with a message.

1.0.9
============

* Release date: 2026/10/21
* Changes and new features:

  * Validate MOCAGE output during configuration parsing: reject multiple vertical levels because MOCAGE
    only supports surface emissions. Report a clear error requesting a single level in ``vertical_description``
    (either a comma-separated list or a CSV file), before processing emissions.

1.0.8
============

* Release date: 2026/10/20
* Changes and new features:

  * Chemical speciation:

    * Added a CSV report of speciation warnings and errors alongside the NetCDF output, replacing .nc with
      _speciation_issues.csv,
      generated independently of debug mode and written only by the MPI master. Includes severity,
      original profiles/components, affected percentages/emissions and conversion impact.
      Detected issues are saved before validation errors stop execution; the log shows a summary and report path.
    * Retained components without CAS or molecular weight in profile composition totals, with warnings
      identifying the excluded component and weight; conversion uses only resolved components without normalization.
    * Changed unselected pollutants in speciation profiles from an error to a warning; their rows are
      excluded from validation and conversion factors while original profile identifiers remain valid.
    * Added warnings for original pollutants missing from used speciation profiles, including the unconverted
      annual emission amount and affected record count for sector and point-source inventories.
    * Removed the obsolete TOG-to-VOC ratio from profile inputs, validation and conversion calculations.
      Profiles no longer require this column; additional columns in existing CSV files are ignored.
    * Improved documentation of input CSV formats, conversion factors, validation rules and DataFrame side effects.
    * Corrected return type annotations and documented optional point-source inventories.
    * Fixed missing-profile error reporting for inventories without identifying columns and for non-string profile IDs.
    * Added regression tests covering multiple profiles, component aggregation, profile deduplication and point sources.

1.0.7
============

* Release date: 2026/10/19
* Changes and new features:

  * geodesic lengths for vector line proxies (``shapefile_line``). Calculate weights from
    physical lengths in metres on the CRS ellipsoid instead of geometry lengths in degrees for geographic CRS.
    Equal angular lengths do not represent equal physical distances: this corrects the relative underweighting
    of north-south lines and overweighting of east-west lines at latitudes such as Catalonia's. No hardcoded
    regional projection is required. Line intersection fractions also use geodesic lengths.
  * Calculate polygon shapefile intersection fractions from geodesic areas rather than coordinate-space areas.
  * Add shared geographical utilities for geodesic lengths, polygon areas and intersection fractions, including
    multipart geometries, polygon holes and empty geometries, with NumPy-style documentation.
  * Raster proxies retain their native CRS. NUTS and destination grid polygons are transformed to the raster CRS
    for intersections, without resampling raster values or selecting a regional projection.
  * Projected raster areas and intersection fractions use planar measurements in the native CRS, with area units
    converted to square metres. Geographic rasters retain geodesic area measurements. Categorical proxy weights
    use these physical areas rather than square degrees.
  * Make numerical raster polygonization deterministic by replacing random decimal offsets and float32 conversion
    with unique integer pixel labels. Each valid numerical pixel remains a separate polygon, including neighbouring
    pixels with equal values, while preserving the existing integer truncation of weights.
  * Numerical categorized proxies retain individual numerical pixels selected by an aligned category raster.
    Validate matching CRS, transforms and raster dimensions, and exclude masked pixels from both inputs.
  * Unify raster reading around a global integer window partitioned into non-overlapping MPI column windows.
    Retain one additional column and row as a precaution, clip the global window to the raster extent, and handle
    empty windows and ranks with no valid pixels.
  * Preserve grouping of contiguous selected categorical pixels within each window to reduce polygon counts.
    Respect raster nodata masks when selecting categories.
  * Intersect raster source polygons with all overlapping cells from the global auxiliary grid, independently of
    MPI destination ownership. Aggregate weights and distribute them to the processes owning each destination FID.
  * Transform raster bounds to the auxiliary grid CRS before selecting grid polygons from disk.
  * Add regression tests for geodesic measurements, native-CRS intersections, deterministic numerical pixels,
    category masks, raster alignment and weight conservation with 1, 8 and 32 simulated MPI partitions.

1.0.6
============

* Release date: 2026/08/04
* Changes and new features:

  * Added support for configuring ``vertical_description`` either as a comma-separated list of layer heights or as a
    CSV file using the historical HERMES_GR format with a ``height_magl`` column.

1.0.5
============

* Release date: 2026/08/03
* Changes and new features:

  * Added consistency checks for auxiliary grid and proxy files, validating that existing ``grid.nc``,
    ``grid.geojson`` and ``<sector>_proxy.nc`` files match the current domain/grid configuration before they are
    reused.

1.0.4
============

* Release date: 2026/06/09
* Changes and new features:

  * Updating NES to v1.2.5, wich includes:
   
    * Fix use of character array in WRF_CHEM output format.

1.0.3
============

* Release date: 2026/05/04
* Changes and new features:

  * Fixed bug when selecting region that does not contain the categorical raster mentioned in the proxies.
  * Improved Temporal tests and holidays usage
  * Removed unnecessary if DEBUG
  * Fixed DEBUG mode with log_level == 5

1.0.1
============

* Release date: 2026/02/18
* Changes and new features:

  * Fixed an issue where point sources located on the boundary between two grid cells were sometimes duplicated.
  * Added a filter to avoid using non-domain regions (e.g., Canary Island over a Iberian Peninsula domain)

1.0.0
============

* Release date: 2026/01/15
* Changes and new features:

  * First official release of HERMESv3_Δ

  * Configuration:

    * Unified INI configuration file with clearly defined sections
    * Support for command-line overrides
    * Region and time zone information via YAML files
    * Validation of all input parameters and mandatory arguments

  * Input formats:

    * CSV:

      * SEIE and Point Source emissions
      * Global attributes for CMAQ/WRF-CHEM/CHIMERE outputs

    * YAML:

      * Temporal profiles (monthly, weekly, daily, hourly)
      * Vertical profiles
      * Region metadata

    * GeoJSON / Shapefile:

      * Spatial proxies
      * Administrative boundaries (NUTS2)

    * Raster:

      * Numeric and categorical proxies (population, land use, crops, urbanisation)

  * Temporal disaggregation:

    * Year → Month → Week → Hour
    * Year → Day → Hour
    * Leap-year consistency for daily profiles
    * Holiday detection by region and optional weekend substitution
    * Time zone conversion from local time to UTC

  * Spatial disaggregation:

    * Dynamic mapping of SEIE proxies to YAML-defined proxies
    * Normalization by Region level
    * Support for raster and vector proxies
    * Weight calculation and validation per region

  * Subdomain workflow:

    * Independent subdomain execution with precomputed proxy totals
    * Preservation of mass balance across domains
    * Automated reuse of parent-domain proxy totals

  * Output:

    * Supported models:

      * MONARCH
      * CMAQ
      * WRF-CHEM
      * MOCAGE
      * CHIMERE
      * DEFAULT (generic NetCDF CF-Compliant)

    * Optional global attributes CSV for CMAQ/WRF-CHEM/CHIMERE
    * Consistent variable and dimension naming
    * Output in NetCDF format compliant with CF conventions

  * Profiles:

    * Temporal:

      * Monthly, weekly, daily, and hourly profiles
      * Sum validation (12, 7, 365/366, 24)

    * Vertical:

      * Normalized fraction profiles (sum = 1)

    * Proxy linkage:

      * Spatial proxies dynamically assigned by sector

  * Validation:

    * Automatic consistency checks for:

      * Missing or conflicting temporal profiles
      * Proxy and profile combinations
      * File existence and data type

    * Error messages include index name and context

  * Compatibility:

    * Compatible with MONARCH, CMAQ, WRF-CHEM, CHIMERE,  and MOCAGE CTMs
    * Region-based holiday detection with `holidays` library
    * Support for projections handled by GDAL/pyproj
