from os import path, makedirs
from typing import Dict, Union
from numpy import sum, arange, array, ndarray, any, isnan, nan_to_num, zeros, int16, float64
from pandas import DataFrame, Series, Index, read_csv, concat
from geopandas import GeoDataFrame, sjoin, read_file
from shapely.affinity import translate
from nes import open_netcdf, Nes
from hermes_d import DEBUG
from hermes_d.config import log_message, get_time_stamp, record_time, precision
from hermes_d.modules.proxies import Proxy
from hermes_d.modules.speciation import SpeciationDelta
from hermes_d.modules.temporal import TemporalDelta
from hermes_d.modules.vertical import VerticalDelta
from gc import collect

REGION_COLUMN_ID = 'NUTS_ID'


class Sector(object):
    """
    Class that defines a sector for environmental analysis.

    Attributes
    ----------
    name : str
        Sector name.
    grid : Nes
        Destination grid.
    clip : GeoDataFrame
        Region where emissions are encapsulated.
    only_horizontal : bool, optional
        Indicates if only horizontal interpolation is desired. Default is False.
    data_paths : dict of str, optional
        Dictionary with paths to data sources. Mandatory keys: "SEIE". Default is None.
    proxies_paths : dict of str, optional
        Dictionary with paths to proxy sources. Mandatory keys: "NUTS2_shp". Default is None.
    profiles_paths : dict of str, optional
        Dictionary with paths to profile sources. Default is None.
    pollutants : list of str
        List of pollutants for the sector.
    sector_info : DataFrame
        Table with the needed sector information.
    proxies : Nes
        Proxies in the NES format.

    Raises
    ------
    ValueError
        If mandatory paths for data or proxies are not provided.
    """

    def __init__(self, sector_name, grid, clip, only_horizontal=False, data_paths=None, proxies_paths=None,
                 profiles_paths=None, pollutants=None, time_step_list=None, holidays=False, consistency_sei=True,
                 mass_emissions=True, daily_consistency=True):
        """
         Initialize the Sector class.
         1. Read SEIE and LPS data.
         2. Speciation: Create mapping and update P_spec.
         3. Vertical: Create mapping and update P_vert.
         4. Temporal: Create mapping and update P_month, P_week, P_day, P_hour.
         4. Proxy

         Parameters
         ----------
         sector_name : str
             Sector name.

         grid : Nes
             Destination grid.

         clip : GeoDataFrame
             Region where emissions are encapsulated.

         only_horizontal : bool, optional
             Indicates if only horizontal interpolation is desired. Default: False.

         data_paths : dict of str, optional
             Dictionary with paths to data sources. Mandatory keys: "SEIE". Default: None.

         proxies_paths : dict of str, optional
             Dictionary with paths to proxy sources. Mandatory keys: "NUTS2_shp". Default: None.

         profiles_paths : dict of str, optional
             Dictionary with paths to profile sources. Default: None.

        holidays: bool, optional
            Indicates if you want to take into account regional (NUTS2) holidays as Sundays.

        consistency_sei: bool, optional
            Indicates if you want to apply a consistency scaling factor is SEIE emissions are lower that LPS/PRTR ones.

        daily_consistency : bool
            Boolean indicating whether to raise an error if the temporal resolution is
            daily and the sum of the temporal profiles does not match the leap year or non-leap year
            status of the simulated year.

         Raises
         ------
         ValueError
             If mandatory paths for data or proxies are not provided.
         """
        first_time = st_time = get_time_stamp()
        # Input checks
        # profiles_paths
        profiles_paths_keys = ['Hourly_profiles', 'Weekly_profiles', 'Monthly_profiles', 'Daily_profiles',
                               'Speciation_profiles', 'Vertical_profiles', 'individual_speciation_file',
                               'spec_aux_path', 'nuts2_info']
        if not set(profiles_paths.keys()).issubset(profiles_paths_keys):
            keys_not_in_set = set(profiles_paths.keys()) - set(profiles_paths_keys)
            err_msg = "Found unexpected key '" + ",".join(keys_not_in_set) + "' in profiles_paths. " + \
                      "Please check the configuration convention."
            raise KeyError(err_msg)
        if not only_horizontal and 'Vertical_profiles' not in profiles_paths.keys():
            raise KeyError("Vertical distribution is required but 'Vertical_profiles' is not provided in config file.")
        # data_paths
        if not set(data_paths.keys()).issubset(['SEIE', 'Point_Sources']):
            keys_not_in_set = set(data_paths.keys()) - {'SEIE', 'Point_Sources'}
            err_msg = "Found unexpected key '" + ",".join(keys_not_in_set) + "' in data_paths. " + \
                      "Please check the configuration convention."
            raise KeyError(err_msg)
        # proxies_paths
        # TODO An-Chi: I've disabled that check because the options are anything that starts with 'shapefile_point',
        #  'shapefile_polygon', 'shapefile_line' plus the others.
        # profiles_paths_keys = ['shapefile_point', 'shapefile_polygon', 'shapefile_line', 'shapefile_polygon_military',
        #                        'shapefile_polygon_agriculture', 'shapefile_polygon_forestry',
        #                        'shapefile_polygon_domestic', 'shapefile_polygon_international', 'population',
        #                        'population_type', 'land_use', 'NUTS2_shp', 'Proxy_aux_path']
        # if not set(proxies_paths.keys()).issubset(profiles_paths_keys):
        #     keys_not_in_set = set(proxies_paths.keys()) - set(profiles_paths_keys)
        #     err_msg = "Found unexpected key '" + ",".join(keys_not_in_set) + "' in proxies_paths. " + \
        #               "Please check the configuration convention."
        #     raise KeyError(err_msg)

        self.name = sector_name
        self.grid = grid
        self.clip = clip
        self.data_paths = data_paths
        self.proxies_paths = proxies_paths
        self.profiles_paths = profiles_paths
        self.only_horizontal = only_horizontal
        self.time_step_list = time_step_list
        self.mass = mass_emissions

        self.master = self.grid.master
        self.rank = self.grid.rank
        self.comm = self.grid.comm

        log_message("Initialization of {0} sector".format(self.name), level=1)

        if pollutants is None:
            warn_msg = "No pollutants provided. Using default list: " + ",".join(self.get_pollutant_list())
            log_message(warn_msg, level=7)
            self.pollutants = self.get_pollutant_list()
        else:
            self.pollutants = pollutants.split(',')

        record_time(self.name, "Init General", get_time_stamp() - st_time)
        # ===== Read SEIE and point source files =====
        self.nuts2_list = self.get_nuts2_list()
        st_time = get_time_stamp()
        self.sector_info = self.get_sector_info()
        if self.sector_info.empty:
            raise ValueError("The emission inventory file is empty.")
        self.point_sources = self.get_point_sources()
        if self.point_sources is not None:
            self.point_sources_corrections(consistency=consistency_sei)

        log_message(f"Region list: {self.nuts2_list}", level=3)
        record_time(self.name, "Init Read", get_time_stamp() - st_time)

        # ===== Speciation =====
        st_time = get_time_stamp()
        # NOTE: No MPI communication here since the files are not big.
        # If we have individual_speciation_file specified in .ini file, do speciation
        if self.profiles_paths["individual_speciation_file"]:
            self.sector_info, self.point_sources, self.species_list, self.speciation_mapping = SpeciationDelta(
                spec_snap_path=self.profiles_paths["Speciation_profiles"],
                indi_spec_path=self.profiles_paths["individual_speciation_file"],
                spec_aux_path=self.profiles_paths['spec_aux_path'],
                sector_info=self.sector_info,
                point_sources=self.point_sources,
            ).initialize(self.pollutants, sector_name, DEBUG)
        else:
            # Skip speciation and remove P_spec column in seie and lps directly
            log_message("No individual_speciation_file provided. Speciation process is skipped.", level=2)
            self.sector_info = self.sector_info.drop(columns="P_spec")
            if self.point_sources is not None:
                self.point_sources = self.point_sources.drop(columns="P_spec")

        record_time(self.name, "Init Speciation", get_time_stamp() - st_time)
        # ===== Vertical distribution =====
        st_time = get_time_stamp()
        if not only_horizontal:
            # initialize class instance
            self.sector_info, self.vertical_factors = VerticalDelta(profiles_paths=self.profiles_paths,
                                                                    sector_info=self.sector_info,
                                                                    output_layers=self.grid.lev['data']).initialize()

        record_time(self.name, 'Init Vertical', get_time_stamp() - st_time)

        # ===== Temporal distribution =====
        st_time = get_time_stamp()
        if not only_horizontal:
            self.sector_info, self.point_sources, self.temporal_factors = TemporalDelta(
                seie=self.sector_info, point_sources=self.point_sources, profiles_paths=profiles_paths,
                time_step_list=time_step_list, holidays=holidays, daily_consistency=daily_consistency).initialize()
        else:
            # Remove P_hour etc. and set P_temp to 'default' in SEIE and LPS
            self.sector_info = self.sector_info.drop(columns=["P_month", "P_week", "P_day", "P_hour"])
            self.sector_info["P_temp"] = "default"
            if self.point_sources is not None:
                self.point_sources = self.point_sources.drop(columns=["P_month", "P_week", "P_day", "P_hour"])
                self.point_sources["P_temp"] = "default"
        record_time(self.name, 'Init Temporal', get_time_stamp() - st_time)

        st_time = get_time_stamp()
        # ===== Proxies =====
        self.proxies = self.get_proxies(proxies_paths['Proxy_aux_path'])
        record_time(self.name, "Init Horizontal", get_time_stamp() - st_time)

        # ===== Units =====
        # output_model: DEFAULT; only_horizontal: False; individual_speciation_file: None
        st_time = get_time_stamp()
        if self.mass:
            # output_model: DEFAULT
            if not self.only_horizontal:
                # only_horizontal: False
                if self.profiles_paths["individual_speciation_file"] in [None, 'None', 'False', False, 0]:
                    # individual_speciation_file: None
                    for pol_name in self.pollutants:
                        # T to kg
                        self.sector_info[pol_name] *= 1000
                    if self.point_sources is not None:
                        for pol_name in self.pollutants:
                            # T to kg
                            self.point_sources[pol_name] *= 1000
                    pass

        if DEBUG:
            self.evaluate_proxy()

        record_time(self.name, "Init General", get_time_stamp() - st_time)

    def evaluate_proxy(self):
        """
        Evaluates the proxies by summing data values over specified dimensions and gathers the results.

        After calculating the sum of proxies, it checks if each result is within a 1% tolerance of 100%.
        If any proxy deviates beyond this tolerance, a single error message is raised, listing all NUTS2_IDs
        and proxies with issues.

        Returns
        -------
        None
            The function writes the results to a CSV file if `self.master` is True.

        Raises
        ------
        ValueError
            If any proxy value deviates by more than 1% from the expected total (100%), listing all
            problematic proxies and NUTS2_IDs.

        Notes
        -----
        - The function assumes that `self.proxies` contains an attribute `variables` (a dictionary-like structure)
          with proxy data as arrays, and `global_attrs` containing a key `"nuts_2_list"`, which lists regions (NUTS2).
        - If `self.master` is True, the gathered results are saved in a CSV file at the path specified by
          `self.proxies_paths['Proxy_aux_path']`, replacing the `.nc` extension with `.csv`.

        Examples
        --------
        evaluate_proxy()  # Call within an instance where the method is defined
        """

        # Retrieve list of proxy variable names
        proxy_list = list(self.proxies.variables.keys())

        # Retrieve list of NUTS2 region identifiers
        nut_list = self.proxies.global_attrs["nuts_2_list"]

        if isinstance(nut_list, str):
            nut_list = [nut_list]

        # Initialize a DataFrame to store results if `self.master` is True
        if self.master:
            result = DataFrame(columns=proxy_list, index=Index(nut_list, name="NUTS2_ID"))
        else:
            result = None

        # List to store any discrepancies for reporting
        discrepancies = []

        # Loop over each proxy variable
        for proxy_name in proxy_list:
            # Sum the proxy data over the dimensions 0 (time), 2 (lat), and 3 (lon)
            proxy_value = sum(self.proxies.variables[proxy_name]["data"], axis=(0, 2, 3))

            # Gather the summed values from all processes if in a parallel environment
            proxy_value = self.comm.gather(proxy_value, root=0)

            # If `self.master` is True, aggregate the gathered results and store in `result` DataFrame
            if self.master:
                proxy_value = sum(array(proxy_value), axis=0)
                result.loc[:, proxy_name] = proxy_value

                # Identify any values outside the tolerance range (1% from 1, i.e., [0.99, 1.01])
                for nuts_id, value in zip(nut_list, proxy_value):
                    if not (0.99 <= value <= 1.01):
                        discrepancies.append(f"{proxy_name} proxy doesn't sum 100% for the {nuts_id} NUTS2_ID")

        # Save the results to a CSV file if `self.master` is True
        if self.master:
            # Define the output path by replacing '.nc' with '.csv' in the given path
            proxy_eval_path = self.proxies_paths['Proxy_aux_path'].replace('.nc', '.csv')

            # Write the DataFrame to a CSV file
            result.to_csv(proxy_eval_path)

            # Print a message indicating where the output file is saved
            log_message(f"Proxy evaluation at {proxy_eval_path}")

            # Raise an error if there are discrepancies
            if discrepancies:
                log_message("\n".join(discrepancies) + f"\nCheck the evaluation file at {proxy_eval_path}")

        return None

    def get_nuts2_list(self) -> list:
        """
        Get the list of NUTS2 codes.

        Returns
        -------
        list of str
            List of NUTS2 codes.
        """
        local_grid = self.grid.shapefile
        all_grids = self.grid.comm.gather(local_grid, root=0)
        if self.grid.master:
            full_grid = concat(all_grids, ignore_index=True)
            nuts2_info = read_file(self.proxies_paths["NUTS2_shp"])

            nuts2_info = nuts2_info.to_crs(full_grid.crs)

            nuts2_list = nuts2_info.sjoin(full_grid, predicate="intersects")

            nuts2_list = list(nuts2_list[REGION_COLUMN_ID].unique())
        else:
            nuts2_list = None

        # Broadcast the result to all nodes in the MPI communicator
        nuts2_list = self.grid.comm.bcast(nuts2_list, root=0)
        return nuts2_list

    def get_nuts2_list_old(self) -> list:
        """
        Get the list of NUTS2 codes.

        Returns
        -------
        list of str
            List of NUTS2 codes.
        """
        if self.grid.master:
            nuts2_list = sorted(list(set(self.sector_info.index.get_level_values(level='NUTS2_code'))))
        else:
            nuts2_list = None

        # Broadcast the result to all nodes in the MPI communicator
        nuts2_list = self.grid.comm.bcast(nuts2_list, root=0)
        return nuts2_list

    def read_proxy(self, proxy_path: str) -> Nes:
        """
        Open and read the proxy data from the given path.

        Parameters
        ----------
        proxy_path : str
            Path to the existing proxy file.

        Returns
        -------
        Nes
            Proxies loaded from the specified path.

        Raises
        ------
        FileNotFoundError
            If the specified proxy file does not exist.
        """
        log_message("Reading proxy", level=2)

        # Check if the file exists
        if not path.exists(proxy_path):
            raise FileNotFoundError(f"The specified proxy file '{proxy_path}' does not exist.")

        # Open the NetCDF file with parallel I/O using MPI communicator
        nessy = open_netcdf(proxy_path, comm=self.grid.comm, parallel_method=self.grid.parallel_method,
                            balanced=self.grid.balanced)

        # Load the proxy data
        nessy.load()

        log_message("Proxies read from {0}".format(proxy_path), level=3)

        return nessy

    def parse_proxy_name(self, spatial_proxy: str, proxy_code: Union[str, int, None]) -> str:
        """
        Build a normalized proxy name from a spatial proxy type and its code.

        This helper converts different proxy specifications (population, population_type,
        land_use, crop_map, shapefile_*) into a canonical string name used across the
        pipeline. For categorical proxies, category codes are parsed as integers,
        sorted ascending, and joined with underscores to ensure stable, reproducible names.

        Parameters
        ----------
        spatial_proxy : str
            Spatial proxy identifier. Supported values are:
            - 'population'
            - 'population_type'
            - 'land_use'
            - 'crop_map'
            - Any string starting with 'shapefile_' (e.g., 'shapefile_polygon').
        proxy_code : Union[str, int, None]
            Code associated with the proxy:
            - For 'population': ignored (may be None).
            - For 'population_type', 'land_use', 'crop_map': a space-separated list of integers
              (e.g., "11 12 13").
            - For 'shapefile_*': the target attribute value or code to select (string).

        Returns
        -------
        str
            Canonical proxy name, e.g., 'population', 'pop_11_12_13', 'lu_1_2', 'crop_map_30',
            or the raw `proxy_code` for 'shapefile_*' proxies.

        Raises
        ------
        ValueError
            If `spatial_proxy` is not recognized.

        Examples
        --------
        >>> self.parse_proxy_name('population', None)
        'population'
        >>> self.parse_proxy_name('population_type', '11 13 12')
        'pop_11_12_13'
        >>> self.parse_proxy_name('land_use', '2 1')
        'lu_1_2'
        >>> self.parse_proxy_name('shapefile_polygon', '020202_others')
        '020202_others'
        """
        if spatial_proxy == 'population':
            proxy_name = 'population'
        elif spatial_proxy == 'population_type':
            categories = sorted([int(x) for x in proxy_code.split(" ")])
            proxy_name = f"pop_{'_'.join(map(str, categories))}"
        elif spatial_proxy == 'land_use':
            categories = sorted([int(x) for x in proxy_code.split(" ")])
            proxy_name = f"lu_{'_'.join(map(str, categories))}"
        elif spatial_proxy == 'crop_map':
            categories = sorted([int(x) for x in proxy_code.split(" ")])
            proxy_name = f"crop_map_{'_'.join(map(str, categories))}"
        elif spatial_proxy.startswith('shapefile_'):
            proxy_name = proxy_code
        else:
            msg = f"Unknown spatial proxy type '{spatial_proxy}' from '{self.data_paths['SEIE']}'"
            raise ValueError(msg)
        return proxy_name

    def parse_double_proxy_name(
            self,
            spatial_proxy_1: str,
            spatial_proxy_2: str,
            proxy_code_1: Union[str, int, None],
            proxy_code_2: Union[str, int, None],
    ) -> str:
        """
        Build a normalized name for a *double proxy* by combining two single proxy names.

        The final name is the concatenation of the two single-proxy names produced by
        `parse_proxy_name`, joined with an underscore. Category lists are normalized
        (sorted) by `parse_proxy_name`, ensuring stable names even if inputs arrive in
        different orders.

        Parameters
        ----------
        spatial_proxy_1 : str
            Spatial proxy type for the first proxy (see `parse_proxy_name`).
        spatial_proxy_2 : str
            Spatial proxy type for the second proxy (see `parse_proxy_name`).
        proxy_code_1 : Union[str, int, None]
            Code for the first proxy (see `parse_proxy_name`).
        proxy_code_2 : Union[str, int, None]
            Code for the second proxy (see `parse_proxy_name`).

        Returns
        -------
        str
            Canonical double-proxy name, e.g., 'pop_11_12_13_shapefileXYZ',
            'population_lu_1_2', or 'pop_11_12_13_pop_21_22'.

        Raises
        ------
        ValueError
            Propagated from `parse_proxy_name` if any component proxy type is unknown.

        Examples
        --------
        >>> self.parse_double_proxy_name('population_type', 'shapefile_polygon', '13 11 12', '020202_others')
        'pop_11_12_13_020202_others'
        >>> self.parse_double_proxy_name('population', 'land_use', None, '2 1')
        'population_lu_1_2'
        """
        proxy_1_name = self.parse_proxy_name(spatial_proxy_1, proxy_code_1)
        proxy_2_name = self.parse_proxy_name(spatial_proxy_2, proxy_code_2)

        return f"{proxy_1_name}_{proxy_2_name}"

    def parse_proxy_info(self, spatial_proxy: str, proxy_code: Union[str, int, None]) -> dict:
        """
        Parse spatial proxy information and generate a dictionary with relevant details.

        Parameters
        ----------
        spatial_proxy : str
            The type of spatial proxy.
        proxy_code : Union[str, int, None]
            The code associated with the spatial proxy, which could be a string or an integer identifier.

        Returns
        -------
        dict
            A dictionary containing the parsed information for the spatial proxy.

        Raises
        ------
        ValueError
            If an unknown spatial proxy type is encountered.
        """
        if ',' in spatial_proxy:
            # Double proxy
            spatial_proxy_1, spatial_proxy_2 = spatial_proxy.split(',')
            proxy_code_1, proxy_code_2 = proxy_code.split(',')

            proxy_info = {
                'name': self.parse_double_proxy_name(spatial_proxy_1, spatial_proxy_2, proxy_code_1, proxy_code_2),
                'type': 'double',
                'proxies_info': [self.parse_proxy_info(spatial_proxy_1, proxy_code_1),
                                 self.parse_proxy_info(spatial_proxy_2, proxy_code_2)]
            }

        elif spatial_proxy == 'population':
            proxy_info = {
                'name': 'population',
                'type': "raster_num",
                'proxy_data': self.proxies_paths['population'],
            }

        elif spatial_proxy == 'population_type':
            categories = sorted([int(x) for x in proxy_code.split(" ")])
            proxy_info = {
                'name': f"pop_{'_'.join(map(str, categories))}",
                'type': "raster_num_cat",
                'proxy_data': self.proxies_paths['population'],
                'categorized_data': self.proxies_paths['population_type'],
                'categories': categories,
            }

        elif spatial_proxy == 'land_use':
            categories = sorted([int(x) for x in proxy_code.split(" ")])
            proxy_info = {
                'name': f"lu_{'_'.join(map(str, categories))}",
                'type': "raster_cat",
                'proxy_data': self.proxies_paths['land_use'],
                'categories': categories,
            }
        elif spatial_proxy == 'crop_map':
            categories = sorted([int(x) for x in proxy_code.split(" ")])
            proxy_info = {
                'name': f"crop_map_{'_'.join(map(str, categories))}",
                'type': "raster_cat",
                'proxy_data': self.proxies_paths['crop_map'],
                'categories': categories,
            }

        elif spatial_proxy.startswith('shapefile_'):
            proxy_info = {
                'name': proxy_code,
                'type': '_'.join(spatial_proxy.split('_')[:2]),
                'proxy_data': self.proxies_paths[spatial_proxy],
                'column': 'proxy_code',
            }

        else:
            msg = f"Unknown spatial proxy type '{spatial_proxy}' from '{self.data_paths['SEIE']}'"
            raise ValueError(msg)

        return proxy_info

    def get_proxies_info(self) -> list:
        """
        Find the different proxies that must be calculated.

        Returns
        -------
        list of dict
            List with the input information for each proxy.
        """
        # Reset the index of sector_info, select relevant columns, and drop duplicates
        profiles = self.sector_info.reset_index().loc[:, ["spatial_proxy", "proxy_code"]].drop_duplicates()

        result = []

        for i, row in profiles.iterrows():
            # Parsing sector info inputs
            if isinstance(row['spatial_proxy'], str):
                spatial_proxy = row['spatial_proxy'].strip()
            else:
                spatial_proxy = row['spatial_proxy']

            if isinstance(row['proxy_code'], str):
                proxy_code = row['proxy_code'].strip()
            else:
                proxy_code = row['proxy_code']

            proxy_info = self.parse_proxy_info(spatial_proxy, proxy_code)

            result.append(proxy_info)

        return result

    def get_proxy(self, proxy_info: dict) -> Proxy:
        """
        Get the appropriate Proxy object based on the provided proxy information.

        Parameters
        ----------
        proxy_info : dict
            Dictionary containing information about the proxy.

        Returns
        -------
        Proxy
            An instance of the appropriate Proxy subclass based on the provided information.
        """
        from hermes_d.modules import (ProxyRasterNum, ProxyRasterCat, ProxyShpPoint, ProxyShpLine, ProxyShpPoly,
                                      ProxyRasterNumCat, DoubleProxy)
        if proxy_info['type'] == 'raster_num':
            proxy = ProxyRasterNum(
                self.grid, self.clip, self.proxies_paths["NUTS2_shp"], proxy_info["proxy_data"],
                nuts2_list=self.nuts2_list)
        elif proxy_info['type'] == 'raster_cat':
            proxy = ProxyRasterCat(
                self.grid, self.clip, self.proxies_paths["NUTS2_shp"], proxy_info["proxy_data"],
                nuts2_list=self.nuts2_list,
                category_list=proxy_info["categories"])
        elif proxy_info['type'] == 'raster_num_cat':
            proxy = ProxyRasterNumCat(
                self.grid, self.clip, self.proxies_paths["NUTS2_shp"], proxy_info["proxy_data"],
                category_path=proxy_info["categorized_data"], nuts2_list=self.nuts2_list,
                category_list=proxy_info["categories"])
        elif proxy_info['type'] == 'shapefile_point':
            proxy = ProxyShpPoint(
                self.grid, self.clip, self.proxies_paths["NUTS2_shp"], proxy_info["proxy_data"],
                nuts2_list=self.nuts2_list, shp_column_value=proxy_info["name"],
                shp_column_name=proxy_info['column'])
        elif proxy_info['type'] == 'shapefile_line':
            proxy = ProxyShpLine(
                self.grid, self.clip, self.proxies_paths["NUTS2_shp"], proxy_info["proxy_data"],
                nuts2_list=self.nuts2_list, shp_column_value=proxy_info["name"],
                shp_column_name=proxy_info['column'])
        elif proxy_info['type'] == 'shapefile_polygon':
            proxy = ProxyShpPoly(
                self.grid, self.clip, self.proxies_paths["NUTS2_shp"], proxy_info["proxy_data"],
                nuts2_list=self.nuts2_list, shp_column_value=proxy_info["name"],
                shp_column_name=proxy_info['column'])
        elif proxy_info['type'] == 'double':
            proxy = DoubleProxy(self.get_proxy(proxy_info['proxies_info'][0]), proxy_info['proxies_info'][1])
        else:
            raise RuntimeError(f"Unknown proxy type {proxy_info['type']}")
        collect()
        return proxy

    def create_proxy(self, proxy_path: str) -> Nes:
        """
        Create and save the Proxy file.

        Parameters
        ----------
        proxy_path : str
            Path to the proxies file to be created.

        Returns
        -------
        Nes
            Proxies.

        Raises
        ------
        ValueError
            If an unknown proxy type is encountered.

        Notes
        -----
        This method creates a new proxy file based on the specified proxy information.

        """
        log_message("Creating proxies", level=2)
        makedirs(path.dirname(proxy_path), exist_ok=True)
        nessy_proxy = self.grid.copy(copy_vars=True)
        nessy_proxy.set_levels({'data': arange(len(self.nuts2_list), dtype=int16)})
        nessy_proxy.global_attrs["nuts_2_list"] = self.nuts2_list

        proxies_info = self.get_proxies_info()

        for i, proxy_info in enumerate(proxies_info):

            aux_time = get_time_stamp()

            log_message(f"Creating {proxy_info['name']} proxy. {i + 1}/{len(proxies_info)}", level=2)

            # Create proxy info with the src resolution
            proxy = self.get_proxy(proxy_info)

            # Change resolution to dst and add proxy to common NES proxy object
            proxy.add_proxy_to_grid(proxy_info['name'], proxy_path.replace('.nc', '_Totals.csv'), nessy_proxy)

            self.comm.Barrier()

            log_message(f"{proxy_info['name']} proxy done in {get_time_stamp() - aux_time:.3f} s", level=3)
        # Write
        log_message("Writing proxies file", level=2)
        nessy_proxy.to_netcdf(proxy_path, serial=True)
        log_message(f"{proxy_path}", level=3)

        if DEBUG:
            nessy_aux = nessy_proxy.copy(copy_vars=True)
            nessy_aux.sum_axis(axis="Z")
            nessy_aux.to_shapefile(path=proxy_path.replace('.nc', '.geojson'), info=False)

        return nessy_proxy

    def get_proxies(self, proxy_path: str) -> Nes:
        """
        Obtain the proxies.

        Parameters
        ----------
        proxy_path : str
            Path to the proxies auxiliary file.

        Returns
        -------
        Nes
            Proxies
        """
        if path.exists(proxy_path):
            proxy = self.read_proxy(proxy_path)
        else:
            proxy = self.create_proxy(proxy_path)

        return proxy

    @staticmethod
    def get_pollutant_list() -> list:
        """
        Obtain the complete list of pollutants.

        Returns
        -------
        list of str
            List of pollutants to take into account.
        """
        pollutants = ['nox_no2', 'nmvoc', 'sox', 'co', 'nh3', 'pm10', 'pm25', 'bc', 'co2', 'ch4', 'n2o', 'pmc']
        return pollutants

    def get_pollutant_units_mapping(self, file_path: str) -> dict or None:
        """
        Reads the first two rows of a CSV file and returns a dictionary
        mapping pollutant names (first row) to their corresponding units (second row).

        Parameters
        ----------
        file_path : str
            Path to the CSV file.

        Returns
        -------
        dict
            Dictionary where keys are pollutant names and values are their respective units.
        """
        if file_path in [None, 'None', 'False', False, 0]:
            result = {}
            for pol_name in self.pollutants:
                if self.mass:
                    result[pol_name] = 'T.s-1'
                else:
                    result[pol_name] = 'T.m-2.s-1'
        else:
            df = read_csv(file_path, nrows=2, header=None)
            result = dict(zip(df.iloc[0, 2:], df.iloc[1, 2:]))
        if self.only_horizontal:
            for poll_name in result.keys():
                if 's' in result[poll_name] or 'h' in result[poll_name]:
                    result[poll_name] = result[poll_name].replace('h-1', 'year-1')
                    result[poll_name] = result[poll_name].replace('h', 'year')
                    result[poll_name] = result[poll_name].replace('s-1', 'year-1')
                    result[poll_name] = result[poll_name].replace('s', 'year')
        else:
            if self.mass:
                # if output_model: DEFAULT
                for poll_name in result.keys():
                    result[poll_name] = result[poll_name].replace('T', 'kg')

        return result

    def get_seie_column_types(self) -> dict:
        """
        Obtain the column names and types to be read from the data file.

        Returns
        -------
        dict
            Dictionary with the column name as the key and the data type as the value.
        """
        dtypes = {'SNAP_activity': str,
                  'NUTS2_code': str,
                  'CONS': bool,
                  'is_point_source': bool,
                  'spatial_proxy': str,
                  'proxy_code': str,
                  'P_month': str,
                  'P_week': str,
                  'P_day': str,
                  'P_hour': str,
                  'P_spec': str,
                  'P_vert': str}
        for pol_name in self.pollutants:
            dtypes[pol_name] = precision
        return dtypes

    def get_sector_info(self) -> DataFrame or None:
        """
        Obtain the sector information.

        The rows will be filtered by the ones with the "CONS" column activated.

        Returns
        -------
        DataFrame
            Table with the active sector information.
        """
        if self.master:
            column_types = self.get_seie_column_types()
            # To conserve the index_col to be string, we need to separate into two steps
            df = read_csv(self.data_paths['SEIE'], usecols=list(column_types.keys()), dtype=column_types)
            df = df.loc[df['NUTS2_code'].isin(self.nuts2_list)]  # Region filter
            df.set_index(['SNAP_activity', 'NUTS2_code'], inplace=True)

            df = df.loc[df['CONS']]
            df.drop(columns=['CONS'], inplace=True)

            # Remove proxy_code content when population is the spatial_proxy
            df.loc[df['spatial_proxy'] == 'population', 'proxy_code'] = None

        else:
            df = None

        df = self.comm.bcast(df, root=0)
        return df

    def get_point_sources_column_types(self) -> dict:
        """
        Obtain the column names and type to be read from the data file.

        Returns
        -------
        dict
            Dictionary with the column name as key and the data type as value
        """
        dtypes = {
            'Code': str,
            'SNAP_activity': str,
            'NUTS2_code': str,
            'CONS': bool,
            'Longitude': precision,
            'Latitude': precision,
            'height': precision,
            'plume_rise_factor': precision,
            'P_month': str,
            'P_week': str,
            'P_hour': str,
            'P_day': str,
            'P_spec': str,
        }
        for pol_name in self.pollutants:
            dtypes[pol_name] = precision
        return dtypes

    def get_point_sources(self) -> GeoDataFrame or None:
        """
        Obtain the point sources information.

        The rows will be filtered by the ones with the "CONS" column activated.

        Returns
        -------
        GeoDataFrame
            Shapefile with the active sector information
        """
        from shapely.geometry import Point

        if 'Point_Sources' not in self.data_paths.keys():
            return None

        activities = self.sector_info.loc[self.sector_info['is_point_source']].index.get_level_values('SNAP_activity')

        if not len(activities) > 0:
            return None

        log_message("Obtaining Point Sources", level=2)

        if self.master:
            column_types = self.get_point_sources_column_types()
            # To conserve the index_col to be string, we need to separate into two steps
            # Use "latin1" encoding because of the special characters in the LPS file
            df = read_csv(self.data_paths['Point_Sources'], usecols=list(column_types.keys()), dtype=column_types,
                          encoding='latin1')
            df.set_index(['SNAP_activity', 'NUTS2_code'], inplace=True)

            df = df.loc[df['CONS']]
            df.drop(columns=['CONS'], inplace=True)
            df = df.loc[activities, slice(None), :]

            # Check: "height" and "plume rise factor" cannot be NaN
            if any(isnan(df["height"])):
                err_msg = "Code " + ' '.join(df.loc[df["height"].isna(), "Code"]) + " column 'height' is NA."
                raise ValueError(err_msg)

            if any(isnan(df["plume_rise_factor"])):
                err_msg = "Code " + ' '.join(
                    df.loc[df["plume_rise_factor"].isna(), "Code"]) + " column 'plume_rise_factor' is NA."
                raise ValueError(err_msg)
            displacement = 0.00001
            geometry = [Point(xy) for xy in zip(df.Longitude + displacement, df.Latitude + displacement)]
            df.drop(['Longitude', 'Latitude'], axis=1, inplace=True)
            crs = "EPSG:4326"
            df = GeoDataFrame(df, crs=crs, geometry=geometry).sort_index()

        else:
            df = None

        df = self.comm.bcast(df, root=0)
        return df

    def point_sources_corrections(self, consistency) -> None:
        """
        Apply corrections due to point sources.

        This function compares SEIE (Spatial Emission Inventory for Europe) emissions with point source emissions.
        If discrepancies are found, corrections are applied to ensure consistency.

        Parameters
        ----------

        consistency: bool
            Indicates if you want to apply a consistency scaling factor is SEIE emissions are lower that LPS/PRTR ones.

        Raises
        ------
        RuntimeError
            If errors are detected during the correction process.
        """
        tolerance = 0.00001  # 0.001 %

        log_message("Applying corrections due to Point source", level=3)

        if self.master:
            error_messages = ""
            raise_error = False
            aux_ps = self.point_sources.loc[:, self.pollutants].groupby(level=self.point_sources.index.names).sum()

            for (activity, nut_code), row in (
                    self.sector_info.loc[self.sector_info['is_point_source'], self.pollutants].iterrows()):
                for pol_name, seie_value in row.items():
                    try:
                        ps_value = aux_ps.loc[(activity, nut_code), pol_name]
                        diff = seie_value - ps_value
                        if seie_value == 0:
                            if ps_value == 0:
                                diff_per = 0
                            else:
                                diff_per = 1
                        else:
                            diff_per = abs(diff) / seie_value

                        if -tolerance <= diff_per <= tolerance:
                            diff = 0
                    except KeyError:
                        ps_value = None
                        diff = 0

                    if ps_value is None:
                        if seie_value > 0:
                            # SEIE with point source but no Point Source found
                            error_messages += ("\t\t\t\tERROR: No point source found: " +
                                               f"Activity: '{activity}', NUT: '{nut_code}', Pollutant: {pol_name}, " +
                                               f"SEIE value: {seie_value:.3f}\n")
                            # 1. Raise an error message and stop execution
                            raise_error = True
                    elif diff > 0:
                        pass
                        error_messages += ("\t\t\t\tNote: The SEIE emissions are bigger than the point source ones: " +
                                           f"Activity: '{activity}', NUT: '{nut_code}', Pollutant: {pol_name}, " +
                                           f"SEIE value: {seie_value:.3f}, PS value: {ps_value:.3f}\n")
                        # SEIE emissions bigger than Point Source
                        # 1. Substitute SEIE emissions by the difference between SEIE and LPS
                        # 2. Use SEIE and LPS
                        self.sector_info.loc[(activity, nut_code), pol_name] = diff
                    elif diff < 0:
                        if consistency:
                            error_messages += (
                                    "\t\t\t\tWARNING: The SEIE emissions are smaller than the point source ones: " +
                                    f"Activity: '{activity}', NUT: '{nut_code}', Pollutant: {pol_name}, " +
                                    f"SEIE value: {seie_value:.3f}, PS value: {ps_value:.3f}\n")
                            # SEIE emissions smaller than Point Source
                            # 1. Use the SEIE emissions
                            # 2. Compute the fraction from LPS and SEIE
                            fraction = seie_value / ps_value
                            # 3. Multiply the LPS emissions by the fraction
                            self.point_sources.loc[(activity, nut_code), pol_name] *= fraction
                        # 4. Set SEIE emissions to zero
                        self.sector_info.loc[(activity, nut_code), pol_name] = 0
                    else:
                        if seie_value > 0:
                            pass
                            error_messages += ("\t\t\t\tOK: Emissions from SEIE and Point Sources: " +
                                               f"Activity: '{activity}', NUT: '{nut_code}', Pollutant: {pol_name}, " +
                                               f"SEIE value: {seie_value:.3f}, PS value: {ps_value:.3f}\n")
                        # No differences
                        # 1. Use only LPS
                        # 2. Set SEIE emissions to zero
                        self.sector_info.loc[(activity, nut_code), pol_name] = 0

            if raise_error:
                log_message("\n" + error_messages, level=9)

            elif error_messages != "":
                log_message(error_messages, level=7)

        else:
            # Not master
            self.sector_info = None
            self.point_sources = None

        collect()

        self.sector_info = self.comm.bcast(self.sector_info, root=0)
        self.point_sources = self.comm.bcast(self.point_sources, root=0)

        return None

    def allocate_point_sources(self) -> DataFrame or None:
        """
        Allocate point sources to the grid.

        Returns
        -------
        DataFrame or None
            Point sources allocated to the grid.
        """
        if self.point_sources is None:
            return None

        # Slightly shift all points (~10 cm) to ensure they don’t fall exactly on cell borders,
        self.point_sources["geometry"] = self.point_sources.geometry.apply(
            lambda geom: translate(geom, xoff=1e-6, yoff=1e-6)
        )

        # Spatial join with the grid shapefile
        self.point_sources = sjoin(
            self.point_sources.reset_index(), self.grid.shapefile.reset_index(), predicate='within'
        )

        # Drop unnecessary columns
        self.point_sources = DataFrame(
            self.point_sources.drop(columns=['index_right', 'geometry', 'SNAP_activity', 'NUTS2_code']))

        # Group by relevant columns and sum pollutant emissions
        self.point_sources = self.point_sources.groupby(
            ['level', 'FID', 'P_temp']
        )[self.pollutants].sum()

        return self.point_sources

    def update_sector_info(self) -> DataFrame:
        """
        Update the sector information.

        1. Avoid temporal disaggregation when only_horizontal is activated.
        2. Change the 'spatial_proxy' column values to the proxies one.

        Returns
        -------
        DataFrame
            Sector information updated.
        """

        def modify_spatial_proxy(row):
            spatial_proxy = str(row['spatial_proxy']).strip()
            proxy_code = str(row['proxy_code']).strip()

            if ',' in spatial_proxy:
                spatial_proxy_1, spatial_proxy_2 = spatial_proxy.split(',')
                proxy_code_1, proxy_code_2 = proxy_code.split(',')
                return self.parse_double_proxy_name(spatial_proxy_1, spatial_proxy_2, proxy_code_1, proxy_code_2)

            if spatial_proxy == 'population':
                return spatial_proxy
            elif spatial_proxy == 'population_type':
                categories = sorted([int(x) for x in proxy_code.split(" ")])
                return f"pop_{'_'.join(map(str, categories))}"
            elif spatial_proxy == 'land_use':
                categories = sorted([int(x) for x in proxy_code.split(" ")])
                return f"lu_{'_'.join(map(str, categories))}"
            elif spatial_proxy == 'crop_map':
                categories = sorted([int(x) for x in proxy_code.split(" ")])
                return f"crop_map_{'_'.join(map(str, categories))}"
            elif spatial_proxy.startswith('shapefile_'):
                return proxy_code
            else:
                return spatial_proxy

        # TODO: P_vert part should be done already in vertical distribution part.
        #       Consider doing this in temporal distribution
        # Disabling temporal distribution when only horizontal is activated
        if self.only_horizontal:
            self.sector_info.loc[:, ['P_temp', 'P_vert']] = 'default'
            if self.point_sources is not None:
                self.point_sources.loc[:, ['P_temp', 'P_vert']] = 'default'

        # Apply the custom function to modify the 'spatial_proxy' column
        self.sector_info['spatial_proxy'] = self.sector_info.apply(modify_spatial_proxy, axis=1)

        self.sector_info.drop(columns='proxy_code', inplace=True)

        # Change all NaNs in P_* to "default"
        # TODO: Is it needed?
        self.sector_info.loc[:, ['P_temp']] = self.sector_info.loc[:, ['P_temp']].fillna("default")

        # Group emission by NUTS2, Proxy, Level and temporal distribution
        self.sector_info = self.sector_info.reset_index().groupby(
            ['NUTS2_code', 'spatial_proxy', 'level', 'P_temp'])[self.pollutants].sum()
        if self.point_sources is not None:
            self.point_sources = self.point_sources.reset_index().set_index(
                ['SNAP_activity', 'NUTS2_code', 'level', 'P_temp'])[self.pollutants + ['geometry']]

        return self.sector_info

    def prepare_lazy_output_emissions(self) -> dict:
        """
        Prepare the output file with the metadata (and no data) to create the empty file.

        Returns
        -------
        dict
            Final output variables in NES format without data.
        """
        lazy_vars = {}
        pollutants_mapping = self.get_pollutant_units_mapping(self.profiles_paths["individual_speciation_file"])

        for var_name, units in pollutants_mapping.items():
            lazy_vars[var_name] = {
                'data': None,
                'units': units,
                'long_name': var_name,
                'dtype': precision
            }

        return lazy_vars

    def get_nuts2_list_from_proxy(self) -> list:
        """
        Retrieve the NUTS-2 list from the global attributes of the proxies.

        Returns
        -------
        List
            A list containing NUTS-2 ID information.

        Note
        ----
        This function assumes the existence of a 'proxies' attribute with a 'global_attrs'
        dictionary, where 'nuts_2_list' is a key associated with a list of NUTS-2 information.
        """
        return self.proxies.global_attrs['nuts_2_list']

    def distribute_horizontally(self, seie_emissions: DataFrame or None,
                                point_source_emissions: DataFrame or None) -> dict:
        """
        Distribute emissions horizontally based on spatial proxies.

        Parameters
        ----------
        seie_emissions : DataFrame or None
            Emissions distributed temporally, with a MultiIndex containing the NUTS2 identifier, the
            proxy_code for the spatial distribution, and the vertical level index.
        point_source_emissions : DataFrame or None
            Point source emissions, distributed temporally, structured similarly to seie_emissions.

        Returns
        -------
        dict
            A dictionary where the keys are pollutant names and the values are dictionaries
            containing the key 'data' with a 4D numpy ndarray. The first dimension is time (1),
            the second is the level, and the last two dimensions correspond to the Y and X spatial grid.
        """
        results = {pollutant: {'data': zeros((1, len(self.grid.lev['data']), self.grid.lat['data'].shape[0],
                                              self.grid.lon['data'].shape[-1]))}
                   for pollutant in self.pollutants}

        # Step 1: Distribute SEIE emissions
        for (nuts2_id, proxy_name, level), emissions_group in seie_emissions.groupby(
                level=["NUTS2_code", "spatial_proxy", "level"]):
            # Get the proxy data for the given NUTS2 and proxy_name
            try:
                nut2_idx = self.get_nuts2_list_from_proxy().index(nuts2_id)
                proxy_data = self.proxies.variables[proxy_name]['data'][0, nut2_idx]
            except Exception as e:
                print("EXCEPTION!!! ", e)
                # Handle missing proxy or NUTS2 data if necessary
                continue

            # Multiply emissions by the proxy and accumulate in the result
            for pollutant in self.pollutants:
                if emissions_group[pollutant].sum() > 0:
                    aux_data = emissions_group[pollutant].values * proxy_data
                    aux_data = nan_to_num(aux_data, nan=0)
                    results[pollutant]['data'][0, level, :, :] += aux_data

        # Step 2: Distribute point source emissions
        if point_source_emissions is not None:
            for level, emissions_group in point_source_emissions.groupby(level="level"):
                for pollutant in self.pollutants:
                    if emissions_group[pollutant].sum() > 0:
                        self.grid.shapefile['aux_var'] = 0.0
                        # Convert numpy array to pandas Series with matching index
                        temp_series = Series(emissions_group[pollutant].values,
                                             index=emissions_group.index.get_level_values('FID'))
                        self.grid.shapefile.loc[emissions_group.index.get_level_values('FID'), 'aux_var'] = temp_series

                        # Reshape and assign to the result
                        results[pollutant]['data'][0, level, :, :] += self.grid.shapefile['aux_var'].to_numpy(
                            dtype=float64).reshape(self.grid.lat['data'].shape[0], self.grid.lon['data'].shape[-1])
                        self.grid.shapefile.drop(columns='aux_var', inplace=True)

        # Synchronize processes if using MPI
        self.comm.Barrier()

        if not self.mass:
            # Kg/s -> Kg/m2·s (Flux emissions)
            for var_name in results.keys():
                results[var_name]['data'] /= self.grid.cell_measures["cell_area"]["data"]

        return results

    def get_4d_emissions(self, time_step_idx) -> Dict[str, Dict[str, Union[int, ndarray]]]:
        """
        Calculate and accumulate emissions in 4D format.

        This function computes the 4D emissions data by accumulating the contributions from different
        temporal and/or level profiles. The result is returned as a dictionary with variables in the
        NES.variables format.

        Parameters
        ----------
        time_step_idx : int
            Index of the time step

        Returns
        -------
        dict
            A dictionary with the 4D emissions data in the NES.variables format. Each variable is represented
            as a nested dictionary containing the 'data' key with a NumPy array representing the emission's data.

        Notes
        -----
        The function performs the accumulation based on the specified temporal and/or level profiles.
        """
        # 1. Distribute emissions temporally
        st_time = get_time_stamp()
        log_message(f"{self.name}: Temporal distribution", level=3)
        seie_emissions = self.distribute_temporally(self.sector_info, time_step_idx)
        if DEBUG:
            if self.master and isinstance(seie_emissions, DataFrame):
                seie_emissions.to_csv(self.proxies_paths["Proxy_aux_path"].replace(
                    '_proxy.nc', f'_SEIE_{self.time_step_list[time_step_idx].strftime("%Y%m%d_%H")}.csv'))
        point_source_emissions = self.distribute_temporally(self.point_sources, time_step_idx)
        if DEBUG:
            if isinstance(point_source_emissions, DataFrame):
                point_source_emissions.to_csv(self.proxies_paths["Proxy_aux_path"].replace(
                    '_proxy.nc',
                    f'_LPS_{self.time_step_list[time_step_idx].strftime("%Y%m%d_%H")}_{str(self.rank).zfill(3)}.csv'))
        if DEBUG:
            record_time(f"{self.name}", "Temporal", get_time_stamp() - st_time)

        # 2. Distribute emissions spatially
        st_time = get_time_stamp()
        log_message(f"{self.name}: Spatial distribution", level=3)
        result = self.distribute_horizontally(seie_emissions, point_source_emissions)

        if DEBUG:
            record_time(f"{self.name}", "Horizontal", get_time_stamp() - st_time)

        pollutants_mapping = self.get_pollutant_units_mapping(self.profiles_paths["individual_speciation_file"])

        # Adding units info
        for poll_name, units in pollutants_mapping.items():
            result[poll_name]['units'] = units

        return result

    def distribute_temporally(self, emissions: DataFrame, time_step_idx: int) -> DataFrame | None:
        """
        Distribute annual emissions to hourly emissions based on temporal profiles.

        This function takes a DataFrame with annual emissions and uses corresponding temporal
        profiles to distribute the emissions across hourly time steps. The function handles
        MultiIndex emissions tables, where 'P_temp' is used to map the appropriate temporal profile.
        The resulting DataFrame maintains the original MultiIndex structure except for the 'P_temp'
        level, which is removed.

        Parameters
        ----------
        emissions : DataFrame
            Table containing annual emissions with a MultiIndex that includes 'P_temp'.
            Columns represent different emission pollutants.
        time_step_idx : int
            Index of the time step to select the corresponding temporal profile from a
            predefined set of hourly profiles.

        Returns
        -------
        DataFrame | None
            A DataFrame with the emissions distributed to hourly values, maintaining the original
            MultiIndex structure but removing the 'P_temp' level. If there are duplicated indices
            after removing 'P_temp', their values will be summed. If there are no emissions,
            the function returns None.

        Steps
        -----
        1. **Copy the emissions DataFrame**:
           The input DataFrame is copied to avoid modifying the original data.

        2. **Extract the temporal profile for the given time step**:
           Based on the provided `time_step_idx`, the corresponding hourly temporal profile is extracted
           from the set of profiles.

        3. **Map the temporal profile to the 'P_temp' level**:
           Using `DataFrame.index.get_level_values()` and `Series.map()`, the temporal profile values are aligned
           with the 'P_temp' level of the emissions table.

        4. **Multiply emissions by the mapped temporal profile**:
           The emission values are multiplied by the corresponding temporal profile values, scaling
           the annual emissions to hourly values.

        5. **Remove the 'P_temp' level from the MultiIndex**:
           After applying the temporal profile, the 'P_temp' level is dropped from the MultiIndex, leaving
           the rest of the index unchanged.

        6. **Group and sum duplicate indices**:
           In cases where removing 'P_temp' results in duplicate indices, emissions are grouped by the remaining
           levels of the MultiIndex, and their values are summed.

        Notes
        -----
        - This function assumes that the temporal profiles are preloaded and accessible via `self.yr_to_hr_profiles`.
        - The function handles cases where there are no point source emissions by returning `None`.
        """
        if emissions is None:
            # No point source emissions
            return None

        # Step 1: Make a copy of the emissions DataFrame
        emissions_copy = emissions.copy()

        if self.only_horizontal:
            # Step 5: Remove the 'P_temp' level from the MultiIndex
            if 'P_temp' in emissions.index.names:
                emissions_copy = emissions_copy.droplevel('P_temp')

            # Step 6: Group by the remaining MultiIndex levels and sum values where indices are duplicated
            emissions_copy = emissions_copy.groupby(level=emissions_copy.index.names).sum()
            return emissions_copy

        # Step 2: Extract the corresponding temporal profile for each 'P_temp'
        time_profile = self.temporal_factors.iloc[:, time_step_idx]

        # Step 3: Use Series.map() to align the 'P_temp' with the correct profile values
        profile_mapping = emissions_copy.index.get_level_values('P_temp').map(time_profile)

        # Step 4: Multiply emissions by the corresponding mapped temporal profile
        emissions_copy = emissions_copy.mul(profile_mapping, axis=0)

        # Step 5: Remove the 'P_temp' level from the MultiIndex
        emissions_copy.index = emissions_copy.index.droplevel('P_temp')

        # Step 6: Group by the remaining MultiIndex levels and sum values where indices are duplicated
        emissions_copy = emissions_copy.groupby(level=emissions_copy.index.names).sum()

        return emissions_copy
