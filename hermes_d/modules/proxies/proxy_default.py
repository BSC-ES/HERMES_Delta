from hermes_d.config import log_message, precision
from geopandas import sjoin, read_file, GeoDataFrame
from pandas import DataFrame, Index, concat, read_csv
from numpy import array, array_split, zeros, uint32, concatenate
from mpi4py import MPI
from nes import Nes
from typing import Union, List, Optional
from shapely.geometry import box
import os


class Proxy(object):
    """
    Represents a Proxy for HERMESv3_Delta.

    Attributes
    ----------
    comm : MPI.Comm
        MPI communicator extracted from the grid parameter.
    master : bool
        Indicates if the current process is the MPI master process.
    rank : int
        Rank of the MPI process extracted from the grid parameter.
    size : int
        Quantity of processes involved extracted from the grid parameter.
    grid : Nes
        An instance of the Nes class representing the grid.
    clip : GeoDataFrame
        A GeoDataFrame representing the clip region.
    src_path : str
        Path to the source file containing the proxy data.
    nuts2_list : List[str]
        List of NUTS2 codes to be used for the analysis.
    nuts2_shp_name : str
        Name of the column that contains the NUTS2 codes.
    nuts2_shp_path : GeoDataFrame or str
        NUTS2 shapefile or path to it, extracted from the nuts2_shp parameter.
    nuts2_shp : GeoDataFrame
        NUTS2 shapefile extracted using get_nut2_shapefile method.
    src_shp : GeoDataFrame
        Proxy shapefile extracted using the get_src_shp method.
    """

    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str], src_path: str,
                 nuts2_list: List[str], clip_nuts2=False):
        """
        Initialize the Proxy class with the necessary parameters.

        Parameters
        ----------
        grid : Nes
            An instance of the Nes class representing the grid.
        clip : GeoDataFrame
            A GeoDataFrame representing the clip region.
        nuts2_shp : GeoDataFrame or str
            NUTS2 shapefile or path to it. It can be either a GeoDataFrame containing
            NUTS2 geometries or a string representing the path to the NUTS2 shapefile.
        src_path : str
            Path to the source file containing the proxy data.
        nuts2_list : List[str]
            List of NUTS2 codes to be used for the analysis.
        """
        self.comm = grid.comm
        self.master = grid.master
        self.rank = grid.rank
        self.size = grid.size

        self.grid = grid
        self.clip = clip
        self.src_path = src_path
        self.nuts2_list = nuts2_list

        self.nuts2_shp_name = 'NUTS_ID'
        self.nuts2_shp_path = nuts2_shp
        self.nuts2_shp = self.get_nut2_shapefile(nuts2_shp, var_list=[self.nuts2_shp_name], clip=clip_nuts2)

        self.src_shp = self.get_src_shp(src_path)

    def get_nut2_shapefile(self, nuts2_shp: Union[GeoDataFrame, str], var_list: Optional[List[str]] = None,
                           clip: bool = True) -> GeoDataFrame:
        """
        Prepare the external shapefile.

        1. Read if it is not already read
        2. Filter variables list
        3. Standardize projections

        Parameters
        ----------
        nuts2_shp : GeoDataFrame or str
            External shapefile or path to it.
        var_list : List[str], optional
            External shapefile variables to be computed.
        clip : bool, optional
            Flag indicating whether to apply clipping during reading.

        Returns
        -------
        GeoDataFrame
            External shapefile.

        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.

        Notes
        -----
        - If nuts2_shp is a string, the function reads the external shapefile and includes specified variables.
        - If nuts2_shp is a GeoDataFrame, it assumes the shapefile is already read.
        - The function resets the index if nuts2_shp is a GeoDataFrame.
        - The function standardizes the projection to match the grid shapefile's CRS.
        - It filters the shapefile to include only NUTS2 regions in nuts2_list.
        """
        if var_list is None:
            var_list = []

        if isinstance(nuts2_shp, str):
            # Reading external shapefile
            log_message("Reading NUTS2 shapefile", level=3)
            try:
                if clip:
                    nuts2_shp = read_file(nuts2_shp, bbox=self.clip)

                else:
                    nuts2_shp = read_file(nuts2_shp)
                nuts2_shp = nuts2_shp[var_list + ["geometry"]]
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {nuts2_shp}")

        else:
            log_message("NUTS2 shapefile already read.", level=3)
            nuts2_shp.reset_index(inplace=True)

        if var_list is not None:
            nuts2_shp = nuts2_shp.loc[:, var_list + ['geometry']]

        self.grid.comm.Barrier()
        log_message("NUTS2 shapefile done.", level=3)

        # Standardizing projection
        nuts2_shp = nuts2_shp.to_crs(self.grid.shapefile.crs)

        if clip:
            nuts2_shp = nuts2_shp.loc[nuts2_shp[self.nuts2_shp_name].isin(self.nuts2_list)]

        nuts2_shp.set_index(self.nuts2_shp_name, inplace=True)

        return nuts2_shp

    def get_src_shp(self, path: str) -> GeoDataFrame:
        """
        Obtain the proxy shapefile from the given source path.

        Parameters
        ----------
        path : str
            Path to the source proxy file.

        Returns
        -------
        GeoDataFrame
            Proxy shapefile.

        Notes
        -----
        - The function logs the process of getting data from the specified path.
        - It reads the source proxy file using the read_src_proxy method.
        - Adds NUTS_code and FID to the source shapefile using the add_info_to_src_shp method.
        - Summarizes data by NUT and FID using the get_proxy_by_nut_and_fid method.
        - If the resulting shapefile is empty, it creates an empty DataFrame with required columns.
        - Normalizes the shapefile by NUTS2 regions using the normalize_by_nut method.
        """
        log_message(f"Getting data from {path}.", level=3)
        src_shp = self.read_src_proxy(path)
        self.comm.Barrier()

        log_message("Adding NUTS_code and FID.", level=3)
        src_shp = self.add_info_to_src_shp(src_shp)

        self.comm.Barrier()

        return src_shp

    def get_proxy_by_nut_and_fid(self, src_shp: DataFrame) -> DataFrame:
        """
        Sum the weights that go to the same FID and NUTS2.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source proxy shapefile.

        Returns
        -------
        DataFrame
            Proxy shapefile aggregated by NUTS2 and FID.

        Notes
        -----
        - The function groups the source proxy shapefile by NUTS2 and FID using the 'groupby' method.
        - It aggregates the weights, summing up values for the same NUTS2 and FID.
        """
        src_shp = src_shp.groupby([self.nuts2_shp_name, 'FID']).sum()
        return src_shp

    def normalize_by_nut(self, src_shp: GeoDataFrame or DataFrame,
                         proxy_name: str, totals_path) -> GeoDataFrame or DataFrame:
        """
        Normalize the weight to be 1 if summing all the NUTS2 region weights.

        Parameters
        ----------
        src_shp : GeoDataFrame or DataFrame
            Proxy shapefile to normalize.
        proxy_name: str
            Name of the proxy.

        Returns
        -------
        GeoDataFrame or DataFrame
            Normalized proxy by NUTS2 regions.

        Notes
        -----
        - The function obtains the partial sum of weights by NUTS2 regions using the get_partial_weight_by_nut method.
        - Normalizes the weights in the source proxy shapefile by dividing each value by the corresponding
           NUTS2 region's total weight.
        - The result is a GeoDataFrame with normalized weights by NUTS2 regions.
        """
        # Get total sum by NUTS2
        by_nut = self.get_total_weight_by_nut(src_shp, proxy_name, totals_path)

        log_message(f"Total weight by nut (before normalizing):\n {by_nut}", level=5)

        try:
            src_shp = src_shp / by_nut
        except TypeError as e:
            print(f"Rank: {self.rank}, SrcShp: {src_shp}")
            print(f"Rank: {self.rank}, by_nut: {by_nut}")
            print(e)
            self.comm.Abort(1)

        return src_shp

    def read_src_proxy(self, path: str, clip: Optional[GeoDataFrame] = None) -> GeoDataFrame:
        """
        Read the source proxy and return it as a shapefile.

        Parameters
        ----------
        path : str
            Path to the source proxy file.
        clip : GeoDataFrame, optional
            Region of the raster to read.

        Returns
        -------
        GeoDataFrame
            Source proxy shapefile.

        Notes
        -----
        - This method reads the source proxy file located at the specified path.
        - Optionally, it can apply clipping using the provided GeoDataFrame region.
        - The resulting data is returned as a GeoDataFrame representing the source proxy shapefile.
        """
        pass

    def add_info_to_src_shp(self, src_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Add the NUTS2_codes and FID to the src_shp.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source proxy shapefile.

        Returns
        -------
        GeoDataFrame
            Source shapefile with the NUTS2_ID and FID columns.

        Notes
        -----
        - This method adds NUTS2 codes using the add_nut_codes method.
        - It also adds destination cell FID using the add_fid method.
        """
        # Adding nut CODES
        log_message("Adding NUTS_code.", level=3)
        src_shp = self.add_nut_codes(src_shp, self.nuts2_shp)

        self.comm.Barrier()
        # Adding destination cell FID
        log_message("Adding FIDs.", level=3)
        src_shp = self.add_fid(src_shp, self.grid.shapefile)

        return src_shp

    def add_nut_codes(self, src_shp: GeoDataFrame, nuts2_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the NUTS2 codes and put them in a new column.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Sources shapefile.
        nuts2_shp : GeoDataFrame
            NUTS2 shapefile.

        Returns
        -------
        GeoDataFrame
            Sources shapefile with the NUT2_ID code.
        """
        src_shp = sjoin(src_shp, nuts2_shp.reset_index(), predicate='within')
        src_shp.drop(columns='index_right', inplace=True)

        return src_shp

    def add_fid(self, src_shp: GeoDataFrame, grid_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the FID codes and put them in a new column.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Sources shapefile.
        grid_shp : GeoDataFrame
            Grid shapefile.

        Returns
        -------
        GeoDataFrame
            Sources shapefile with the FID code.
        """
        src_shp = sjoin(src_shp, grid_shp.reset_index(), predicate='within')
        src_shp.drop(columns='index_right', inplace=True)

        return src_shp

    def add_proxy_to_grid(self, proxy_name: str, totals_path: str, proxy_out: Nes) -> Nes:
        """
        Add the proxy to the grid.

        Parameters
        ----------
        proxy_name : str
            Name of the proxy.
        proxy_out : Nes
            A Nes object to add the proxy

        Notes
        -----
        - This method adds a new variable to the grid with the specified proxy_name.
        - The variable is initialized with zeros and has dimensions based on the grid and NUTS2 list.
        - For each NUTS2 region, it extracts data from the source proxy shapefile and assigns it to the grid variable.
        - The data is weighted by the 'weight' column in the source proxy shapefile.
        - The result is stored in the grid variable with the specified proxy_name.
        """
        # Source shpt to grid_shp
        log_message("Summarizing data by NUT and FID.", level=3)

        grid_shp = self.get_proxy_by_nut_and_fid(self.src_shp)

        if len(grid_shp) == 0:
            grid_shp = DataFrame(columns=[self.nuts2_shp_name, 'FID', 'weight']).set_index([self.nuts2_shp_name, 'FID'])
        self.comm.Barrier()

        # Normalizing to 1
        log_message("Normalizing by NUTS2.", level=3)
        grid_shp = self.normalize_by_nut(grid_shp, proxy_name, totals_path)
        self.comm.Barrier()

        # Proxy to NES

        proxy_out.variables[proxy_name] = {
            'data': zeros(
                (1, len(self.nuts2_list), proxy_out.lat['data'].shape[0], proxy_out.lon['data'].shape[-1]),),
        }

        for i_lev, nuts2_id in enumerate(self.nuts2_list):
            if nuts2_id in grid_shp.index.get_level_values(self.nuts2_shp_name):
                aux_data = grid_shp.xs(nuts2_id, level=self.nuts2_shp_name)
                proxy_out.shapefile['aux_var'] = 0.0
                proxy_out.shapefile.loc[aux_data.index, 'aux_var'] = aux_data.loc[:, 'weight']
                proxy_out.variables[proxy_name]['data'][0, i_lev, :] = array(
                    proxy_out.shapefile['aux_var'], dtype=precision).reshape(
                    (proxy_out.lat['data'].shape[0], proxy_out.lon['data'].shape[-1]))
        return proxy_out

    def add_proxy_to_grid_not_used(self, proxy_name: str) -> Nes:
        """
        Add the proxy to the grid. (NOT USED)

        Parameters
        ----------
        proxy_name : str
            Name of the proxy.

        Returns
        -------
        Nes
            NetCDF format data with the proxies

        Notes
        -----
        - This method adds a new variable to the grid with the specified proxy_name.
        - The variable is initialized with zeros and has dimensions based on the grid and NUTS2 list.
        - For each NUTS2 region, it extracts data from the source proxy shapefile and assigns it to the grid variable.
        - The data is weighted by the 'weight' column in the source proxy shapefile.
        - The result is stored in the grid variable with the specified proxy_name.
        """
        proxy = self.grid.copy(copy_vars=False)
        proxy.set_communicator(self.grid.comm)

        proxy.variables[proxy_name] = {
            'data': zeros((1, len(self.nuts2_list), proxy.lat['data'].shape[0], proxy.lon['data'].shape[-1]),),
        }

        for i_lev, nuts2_id in enumerate(self.nuts2_list):
            if nuts2_id in self.src_shp.index.get_level_values(self.nuts2_shp_name):
                aux_data = self.src_shp.xs(nuts2_id, level=self.nuts2_shp_name)
                proxy.shapefile['aux_var'] = 0
                proxy.shapefile.loc[aux_data.index, 'aux_var'] = aux_data.loc[:, 'weight']
                proxy.variables[proxy_name]['data'][0, i_lev, :] = array(
                    proxy.shapefile['aux_var'], dtype=precision).reshape(
                    (proxy.lat['data'].shape[0], proxy.lon['data'].shape[-1]))
        return proxy

    @staticmethod
    def overlay(
            gdf1: GeoDataFrame,
            gdf2: GeoDataFrame,
            gdf1_columns: Optional[List[str]] = None,
            gdf2_columns: Optional[List[str]] = None,
            predicate: Optional[str] = None,
            one_intersection: bool = False,
            weight_vars: Optional[Union[str, List[str]]] = None
    ) -> GeoDataFrame:
        """
        Overlay two GeoDataFrames and merge their information.

        Parameters
        ----------
        gdf1 : GeoDataFrame
            First shapefile with information to merge.
        gdf2 : GeoDataFrame
            Second shapefile with information to merge.
        gdf1_columns : list of str, optional
            Columns from the first GeoDataFrame to include in the result (excluding geometry).
        gdf2_columns : list of str, optional
            Columns from the second GeoDataFrame to include in the result (excluding geometry).
        predicate : str, optional
            Spatial relationship predicate for overlay operation (default is 'intersects').
        one_intersection : bool, optional
            If True, geometries with only one intersection won't be considered as fractions.
        weight_vars : str or list of str, optional
            Variables to be weighted by the portion of area that fits in the FID cell.

        Returns
        -------
        GeoDataFrame
            Unique GeoDataFrame with merged information from both input GeoDataFrames.

        Notes
        -----
        - The resulting GeoDataFrame contains information from both input GeoDataFrames based on the spatial overlay.
        - Spatial overlay is performed using the specified predicate (default is 'intersects').
        - Columns from each GeoDataFrame can be selectively included in the result.
        - If weight_vars is specified, variables are weighted by the portion of area that fits in the FID cell.
        """
        def get_intersect_geom(x):
            if one_intersection or x['_num_intersect'] > 1:
                geom = gdf1.loc[x['gdf1_idx'], 'geometry'].buffer(0).intersection(
                    gdf2.loc[x['gdf2_idx'], 'geometry'].buffer(0))
            else:
                geom = gdf1.loc[x['gdf1_idx'], 'geometry']
            return geom

        def get_intersect_fraction(x):
            return x['geometry'].area / gdf1.loc[x['gdf1_idx'], 'geometry'].area

        # Initialization
        if gdf1_columns is None:
            gdf1_columns = list(gdf1.columns)
            gdf1_columns.remove('geometry')
        if gdf2_columns is None:
            gdf2_columns = list(gdf2.columns)
            gdf2_columns.remove('geometry')
        if predicate is None:
            # Default operation
            predicate = 'intersects'
        if weight_vars is None:
            weight_vars = []
        elif isinstance(weight_vars, str):
            weight_vars = [weight_vars]

        # Reset index
        gdf1 = gdf1.reset_index(drop=True)
        # Both shapefiles must be in the same Coordinate Rate System (CRS)
        gdf2 = gdf2.reset_index(drop=True).to_crs(gdf1.crs)

        # gdf2_idx, gdf1_idx = gdf1.sindex.query_bulk(gdf2.geometry, predicate=predicate)
        gdf2_idx, gdf1_idx = gdf1.sindex.query(gdf2.geometry, predicate=predicate)
        # len(gdf1_idx) == len(gdf2_idx) . This length is the amount of intersections
        # gdf1_idx and gdf2_idx contains the index of each intersection element depending on the source.
        intersection_df = DataFrame(columns=["gdf1_idx", "gdf2_idx"] + gdf1_columns + gdf2_columns)
        if len(gdf2_idx) > 0:
            # Filling with gdf1_data
            intersection_df["gdf1_idx"] = array(gdf1.loc[gdf1_idx].index, dtype=uint32)
            for col_name in gdf1_columns:
                intersection_df[col_name] = gdf1.loc[gdf1_idx, col_name].values
            # Filling with gdf2_data
            intersection_df["gdf2_idx"] = array(gdf2.loc[gdf2_idx].index, dtype=uint32)
            for col_name in gdf2_columns:
                intersection_df[col_name] = gdf2.loc[gdf2_idx, col_name].values

            # Geometry
            if not one_intersection:
                intersection_df['_num_intersect'] = intersection_df.groupby('gdf1_idx')['gdf1_idx'].transform('size')
            intersection_df['geometry'] = intersection_df.apply(get_intersect_geom, axis=1)
            if not one_intersection:
                intersection_df.drop(columns='_num_intersect', inplace=True)

            # Weighting variables
            if len(weight_vars) > 0:
                intersection_df['_fraction'] = array(intersection_df.apply(get_intersect_fraction, axis=1),
                                                     dtype=precision)

                for var_name in weight_vars:
                    intersection_df[var_name] = array(
                        intersection_df[var_name], dtype=precision) * intersection_df['_fraction']
                intersection_df.drop(columns='_fraction', inplace=True)

        else:
            intersection_df['geometry'] = None

        intersection_df = GeoDataFrame(intersection_df, crs=gdf1.crs)
        return intersection_df

    def get_total_weight_by_nut(self, src_shp: GeoDataFrame, proxy_name: str, totals_path: str) -> DataFrame:
        """
        Get the total weight for each NUTS2 region within the clipped domain.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Proxy shapefile to normalize within the clipped domain.
        proxy_name: str
            Name of the proxy.

        Returns
        -------
        GeoDataFrame
            Normalized proxy by NUTS2 regions within the clipped domain.
        """
        if os.path.exists(totals_path):
            total_df = read_csv(totals_path, index_col=0)

            if proxy_name in total_df.columns:
                result = total_df[[proxy_name]]

            else:
                # Calculate and add new column
                partial_sum = self.get_partial_weight_by_nut(src_shp)
                try:
                    total_df[proxy_name] = partial_sum['weight']
                except KeyError:
                    # No proxy in my region
                    total_df[proxy_name] = 0
                    log_message(f"{proxy_name} do not appear in the selected domain. Setting it as a 0.", level=7)
                if self.master:
                    total_df.to_csv(totals_path)
                self.comm.Barrier()
                result = total_df[[proxy_name]]
        else:
            # File does not exist: create it
            partial_sum = self.get_partial_weight_by_nut(src_shp)
            if 'weight' not in partial_sum.columns:
                partial_sum['weight'] = 0
                log_message(f"{proxy_name} do not appear in the selected domain. Setting it as a 0.", level=7)
            partial_sum.rename(columns={'weight': proxy_name}, inplace=True)
            if self.master:
                partial_sum.to_csv(totals_path)
            result = partial_sum[[proxy_name]]

        # Column renamed to weight
        result.columns = ['weight']
        return result

    from pandas import DataFrame, Index, concat

    def get_partial_weight_by_nut(self, src_shp: GeoDataFrame) -> DataFrame:
        """
        Compute total proxy weights by NUTS2 (partial, within the clipped domain),
        and return a DataFrame whose index is *always* `self.nuts2_list`.
        Missing NUTS in the clipped domain are reported as NaN.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Proxy shapefile (already clipped to the domain). Must have a MultiIndex
            (or Index) with a level named `self.nuts2_shp_name`.

        Returns
        -------
        DataFrame
            Rows indexed by `self.nuts2_list` (Index name = `self.nuts2_shp_name`),
            columns are the summed numeric fields from `src_shp`. NUTS not present
            in the clipped domain are filled with NaN.
        """
        # Local partial sum by NUTS2 on each rank
        # (sum() on a GeoDataFrame drops geometry and sums numeric cols)
        try:
            local = src_shp.groupby(level=self.nuts2_shp_name).sum()
        except KeyError as exc:
            raise KeyError(
                f"Index level '{self.nuts2_shp_name}' not found in src_shp.index.names "
                f"(got {src_shp.index.names})."
            ) from exc

        # Gather to rank 0
        gathered = self.comm.gather(local, root=0)

        if self.master:
            # Filter out Nones/empties to avoid concat warnings
            parts = [df for df in gathered if isinstance(df, DataFrame) and not df.empty]

            if parts:
                # Sum across ranks; min_count=1 to keep NaN if a NUTS is entirely missing
                total = (
                    concat(parts, axis=0, sort=False)
                    .groupby(level=self.nuts2_shp_name, sort=False)
                    .sum(min_count=1)
                )
            else:
                # No data at all from any rank
                total = DataFrame()

            # Reindex to the canonical, full NUTS2 list; missing => NaN
            target_index = Index(self.nuts2_list, name=self.nuts2_shp_name)
            # If total is empty, this yields an all-NaN frame with the right index
            total = total.reindex(target_index)

            by_nut = total
        else:
            by_nut = None

        # Broadcast to all ranks
        by_nut = self.comm.bcast(by_nut, root=0)

        log_message(f"By NUT: shape={by_nut.shape}, index_name={by_nut.index.name}, "
                    f"cols={list(by_nut.columns)}", level=5)
        return by_nut

    def balance(self, data, rank=0):
        """
        Balance data across processes.

        This function balances data across processes by gathering, concatenating, splitting,
        and scattering the data.

        Parameters
        ----------
        data : any
            The data to be balanced across processes.
        rank : int, optional
            The rank of the process that will perform the balancing operation.
            Default is 0.

        Returns
        -------
        any
            The balanced data received by all processes. If the calling process is not the root
            process (rank), it receives the balanced data. Otherwise, it returns None.

        Notes
        -----
        - This function balances data across processes by gathering, concatenating, splitting,
          and scattering the data.
        - The input `data` can be any object.
        - The data is gathered from all processes to the root process, concatenated, split into
          chunks, and then scattered back to all processes.
        """
        data = self.comm.gather(data, root=rank)
        if self.comm.Get_rank() == rank:
            data = concat(data)
            data = array_split(data, self.comm.Get_size())
        else:
            data = None

        data = self.comm.scatter(data, root=rank)

        return data

    def balance_geometries(self, data, rank=0):
        """
        Balance geometries across processes.

        This function balances the geometries across processes by gathering, concatenating,
        and scattering the geometries.

        Parameters
        ----------
        data : GeoDataFrame
            The GeoDataFrame containing geometries to be balanced across processes.
        rank : int, optional
            The rank of the process that will perform the balancing operation.
            Default is 0.

        Returns
        -------
        GeoDataFrame or None
            The GeoDataFrame with balanced geometries if the calling process is the root process (rank).
            Otherwise, returns None.

        Notes
        -----
        - This function balances geometries across processes by gathering, concatenating, and scattering them.
        - The input `data` should be a GeoDataFrame containing geometries.
        - The geometries are gathered from all processes to the root process, concatenated, split into chunks,
          and then scattered back to all processes.
        - The returned GeoDataFrame contains balanced geometries if the calling process is the root process (rank),
          otherwise, it returns None.
        """
        # Extract CRS before gathering
        crs = data.crs

        # Convert GeoDataFrame geometry to numpy array
        data_array = data.geometry.values

        # Gather data from all processes to the root process
        gathered_data = self.comm.gather(data_array, root=rank)

        if self.comm.Get_rank() == rank:
            # Concatenate the gathered data
            concatenated_data = concatenate(gathered_data)

            # Split the concatenated data into chunks
            split_data = array_split(concatenated_data, self.comm.Get_size())
        else:
            split_data = None

        # Scatter the split data to all processes
        scattered_data = self.comm.scatter(split_data, root=rank)

        # Convert scattered data back to GeoDataFrame
        if scattered_data is not None:
            scattered_gdf = GeoDataFrame(geometry=scattered_data, crs=crs)
        else:
            scattered_gdf = None

        return scattered_gdf

    def broadcast(self, data, rank=0):
        """
        Broadcast data to all processes.

        This function broadcasts data from one process to all other processes in the communicator.

        Parameters
        ----------
        data : GeoDataFrame or DataFrame
            The data to be broadcasted. This can be any object that is pickleable.
        rank : int, optional
            The rank of the process that will provide the data to be broadcasted.
            Default is 0.

        Returns
        -------
        GeoDataFrame or DataFrame
            The broadcasted data received by all processes. If the calling process is not the root process (rank),
            it receives the broadcasted data. Otherwise, it returns the original data.

        Notes
        -----
        - This function broadcasts data from one process to all other processes in the communicator.
        - If the communicator has only one process, the data remains unchanged.
        - The data to be broadcasted should be pickleable.
        """
        if self.comm.Get_size() == 1:
            # If only one process, no need to broadcast
            data = data
        else:
            data = self.comm.gather(data, root=rank)
            if self.comm.Get_rank() == rank:
                data = [df for df in data if not df.empty]
                try:
                    data = concat(data)
                except ValueError:
                    data = GeoDataFrame()
            else:
                data = None
            data = self.comm.bcast(data, root=rank)

        return data

    def write_shapefile_parallel(self, data, path, driver=None, rank=0):
        """
        Write data to a shapefile in parallel.

        This function gathers data from all processes and writes it to a shapefile or CSV file in parallel.

        Parameters
        ----------
        data : any
            The data to be written to the shapefile. This can be any object that can be handled by the
            `geopandas.GeoDataFrame` class or a similar tabular data structure.
        path : str
            The path to save the shapefile or CSV file.
        driver : str, optional
            The OGR driver used to write the shapefile. If None, the default driver is used.
            Default is None.
        rank : int, optional
            The rank of the process that will perform the writing operation. Only the process with
            the specified rank will write the file. Other processes will return without writing.
            Default is 0.

        Returns
        -------
        bool
            True if the writing operation is successful, otherwise False.

        Notes
        -----
        - This function writes data to a shapefile or CSV file in parallel.
        - The writing operation is performed only by the process with the rank specified by the `rank` parameter.
          Other processes return without performing the writing operation.
        - The `data` parameter should be a GeoDataFrame or a similar tabular data structure.
        - If `driver` is 'csv', the data will be written to a CSV file. Otherwise, the data will be written
          to a shapefile using the specified OGR driver.
        """
        import os
        from copy import deepcopy

        data_aux = deepcopy(data)
        data_aux = self.comm.gather(data_aux, root=rank)

        if self.comm.Get_rank() == rank:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            data_aux = concat(data_aux)
            if driver.lower() == 'csv':
                data_aux.to_csv(path)
            else:
                data_aux.to_file(path, driver=driver)

        self.comm.Barrier()

        return True

    # noinspection PyProtectedMember
    def get_bbox(self, full_domain: bool = True) -> GeoDataFrame:
        """
        Compute the bounding box of the grid.

        Parameters
        ----------
        full_domain : bool, optional
            If True, computes the bounding box over the full domain.
            If False, computes the bounding box only over the grid bounds.
            Default is True.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame representing the bounding box.

        Notes
        -----
        - This function computes the bounding box of the grid.
        - If `full_domain` is True, the bounding box is calculated over the full domain.
          If False, it is calculated only over the grid bounds.
        - The returned GeoDataFrame represents the bounding box.
        """
        if full_domain:
            grid_box = box(self.grid.get_full_longitudes_boundaries()['data'].min(),
                           self.grid.get_full_latitudes_boundaries()['data'].min(),
                           self.grid.get_full_longitudes_boundaries()['data'].max(),
                           self.grid.get_full_latitudes_boundaries()['data'].max())
        else:
            grid_box = box(self.grid.lon_bnds['data'].min(), self.grid.lat_bnds['data'].min(),
                           self.grid.lon_bnds['data'].max(), self.grid.lat_bnds['data'].max())

        bounds_shp = GeoDataFrame(geometry=[grid_box], crs=self.grid.shapefile.crs)

        nuts2_shp = read_file(self.nuts2_shp_path).to_crs(self.grid.shapefile.crs)

        nuts2_shp = nuts2_shp.sjoin(bounds_shp, predicate='intersects')

        nuts2_box = box(nuts2_shp.total_bounds[0], nuts2_shp.total_bounds[1],
                        nuts2_shp.total_bounds[2], nuts2_shp.total_bounds[3])
        bounds_shp = GeoDataFrame(geometry=[nuts2_box], crs=self.grid.shapefile.crs)

        return bounds_shp
