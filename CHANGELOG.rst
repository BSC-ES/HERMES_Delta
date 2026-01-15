============
CHANGELOG
============

.. start-here


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
