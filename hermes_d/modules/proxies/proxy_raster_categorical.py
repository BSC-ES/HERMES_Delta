from hermes_d.modules.proxies import ProxyRaster
from hermes_d.utilities import redistribute_dataframes as redistribute
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds
from rasterio import open as open_raster
from numpy import isin, array, where, uint16, uint32, float64
from shapely.errors import TopologicalError
from geopandas import GeoDataFrame, read_file
from pandas import DataFrame
from nes import Nes
from typing import Union, List, Optional
from hermes_d.config import precision, log_message


class ProxyRasterCat(ProxyRaster):
    """
    Represents a Categorized ProxyRaster for HERMESv3_Delta.

    Attributes
    ----------
    category_list : List[int]
        List of categories of the raster to consider.

    Notes
    -----
    This class represents a ProxyRaster with additional support for categorizing the raster data based on specified
    categories.

    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str], src_path: str,
                 nuts2_list: List[str], category_list: List[int]):
        """
        Initialize the ProxyRasterCat class with the necessary parameters.

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
        category_list : List[int]
            List of categories of the raster to consider.
        """
        if category_list is None or len(category_list) == 0:
            raise ValueError("Unable to create a categorized proxy. No set categories found.")

        self.category_list = array(category_list)

        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list, clip_nuts2=False)

    def read_src_proxy(self, path: str, clip: Optional[GeoDataFrame] = None) -> GeoDataFrame:
        """
        Read the source proxy raster and convert it to a GeoDataFrame.

        Parameters
        ----------
        path : str
            Path to the source proxy raster file.
        clip : GeoDataFrame, optional
            Region of the raster to read. If not provided, the entire raster is considered.

        Returns
        -------
        GeoDataFrame
            Geometries representing the source proxy shapefile.

        """
        window = self.get_rank_window(path)

        with open_raster(path) as src:
            data = src.read(1, window=window)

            # Apply mask: only keep allowed values and categorize as 1
            mask = isin(data, self.category_list)
            data = where(mask, 1, 0)

            data = data.astype(uint16)

            transform = src.window_transform(window)
            if mask.any():
                shapes_gen = shapes(data, mask=mask, transform=transform)
                gdf = GeoDataFrame.from_features(
                    ({'geometry': shape, 'properties': {'weight': value}} for shape, value in shapes_gen),
                    crs=src.crs
                )
                # Error on to_crs function of geopandas that flip lat with lon in the non dict form
                if str(gdf.crs).startswith('EPSG:'):
                    # Error on to_crs function of geopandas that flips lat with lon in the non-dict form
                    gdf.crs = {'init': 'epsg:{0}'.format(str(gdf.crs)[-4:])}

                gdf = gdf.to_crs({'init': 'epsg:4326'})  # 'EPSG:4326'
            else:
                gdf = GeoDataFrame()
        gdf = redistribute(gdf, self.comm)
        self.comm.Barrier()
        return gdf

    def get_total_window(self, path: str) -> Window:
        """
        Calculates the total window of data for the entire dataset based on the bounding box.

        Parameters
        ----------
        path : str
            The file path to the raster data.

        Returns
        -------
        Window
            A rasterio Window object representing the total window of data for the entire dataset.
        """
        # Get Bounding box
        bbox = self.get_bbox(full_domain=True)
        if self.master:
            # Master:
            with open_raster(path) as src:
                # Reproject bounding box
                bbox = bbox.to_crs(src.crs)
                bbox = array(bbox.total_bounds, dtype=float64)

                # Obtain full window
                window = self.fix_window(from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform))
        else:
            window = None

        window = self.comm.bcast(window)

        return window

    # noinspection DuplicatedCode
    def get_partial_window_x(self, total_window: Window) -> Window:
        """
        Calculates the partial window of data for the current process based on its rank (X direction).

        Parameters
        ----------
        total_window : Window
            The total window of data for the entire dataset.

        Returns
        -------
        Window
            A rasterio Window object representing the partial window of data for the current process in X direction.
        """
        # Extract window parameters
        col_off = total_window.col_off
        row_off = total_window.row_off
        width = total_window.width
        height = total_window.height

        # Calculate the base number of columns each rank should handle
        base_columns_per_rank = width // self.size
        # Calculate the number of extra columns that need to be distributed
        extra_columns = width % self.size

        # Distribute the extra columns among the first few processes
        if self.rank < extra_columns:
            # For ranks less than extra_columns, each gets an additional column
            start_col = col_off + self.rank * (base_columns_per_rank + 1)
            local_width = base_columns_per_rank + 1
        else:
            # For the remaining ranks, they get the base number of columns
            start_col = col_off + self.rank * base_columns_per_rank + extra_columns
            local_width = base_columns_per_rank

        # Create the local Window object for each process
        local_window = Window(col_off=start_col, row_off=row_off, width=local_width, height=height)

        return local_window

    # noinspection DuplicatedCode,PyUnreachableCode
    def get_partial_window_y(self, total_window: Window) -> Window:
        """
        Calculates the partial window of data for the current process based on its rank (Y direction).

        Parameters
        ----------
        total_window : Window
            The total window of data for the entire dataset.

        Returns
        -------
        Window
            A rasterio Window object representing the partial window of data for the current process in Y direction.
        """
        raise NotImplementedError("Method not working properly")
        # Extract window parameters
        col_off = total_window.col_off
        row_off = total_window.row_off
        width = total_window.width
        height = total_window.height

        # Calculate the base number of rows each rank should handle
        base_rows_per_rank = height // self.size
        # Calculate the number of extra rows that need to be distributed
        extra_rows = height % self.size

        # Distribute the extra rows among the first few processes
        if self.rank < extra_rows:
            # For ranks less than extra_rows, each gets an additional row
            start_row = row_off + self.rank * (base_rows_per_rank + 1)
            local_height = base_rows_per_rank + 1
        else:
            # For the remaining ranks, they get the base number of rows
            start_row = row_off + self.rank * base_rows_per_rank + extra_rows
            local_height = base_rows_per_rank

        # Create the local Window object for each process
        local_window = Window(col_off=col_off, row_off=start_row, width=width, height=local_height)

        return local_window

    # noinspection PyUnreachableCode
    def get_partial_window_o(self, total_window: Window) -> Window:
        """
        Calculates the partial square window of data for the current process based on its rank.

        Parameters
        ----------
        total_window : Window
            The total window of data for the entire dataset.

        Returns
        -------
        Window
            A rasterio Window object representing the partial square window of data for the current process.
        """
        raise NotImplementedError("Method not working properly")
        # Extract window parameters
        col_off = total_window.col_off
        row_off = total_window.row_off
        width = total_window.width
        height = total_window.height

        # Calculate the grid size (number of blocks in x and y directions)
        grid_size = int(self.size ** 0.5)

        # Handle cases where the grid is not perfect square
        extra_ranks = self.size - grid_size ** 2
        if extra_ranks > 0:
            grid_size += 1

        # Calculate the size of each block
        base_block_width = width // grid_size
        base_block_height = height // grid_size

        extra_width = width % grid_size
        extra_height = height % grid_size

        # Determine the x and y rank position in the grid
        rank_x = self.rank % grid_size
        rank_y = self.rank // grid_size

        # Calculate starting column and row for the current rank
        start_col = col_off + rank_x * base_block_width + min(rank_x, extra_width)
        start_row = row_off + rank_y * base_block_height + min(rank_y, extra_height)

        # Calculate the width and height for the current rank
        if rank_x < extra_width:
            local_width = base_block_width + 1
        else:
            local_width = base_block_width

        if rank_y < extra_height:
            local_height = base_block_height + 1
        else:
            local_height = base_block_height

        # Create the local Window object for each process
        local_window = Window(col_off=start_col, row_off=start_row, width=local_width, height=local_height)

        return local_window

    def get_partial_window(self, total_window: Window) -> Window:
        """
        Calculates the partial window of data for the current process based on the parallel method.

        Parameters
        ----------
        total_window : Window
            The total window of data for the entire dataset.

        Returns
        -------
        Window
            A rasterio Window object representing the partial window of data for the current process.
        """
        window = self.get_partial_window_x(total_window)
        # window = self.get_partial_window_y(total_window)
        # window = self.get_partial_window_o(total_window)
        return window

    def get_rank_window(self, path: str) -> Window:
        """
        Calculates the window of data for the current process based on its rank.

        Parameters
        ----------
        path : str
            The file path to the raster data.

        Returns
        -------
        Window
            A rasterio Window object representing the window of data for the current process.
        """
        window = self.get_total_window(path)
        window = self.get_partial_window(window)

        return window

    def add_nut_codes(self, src_shp: GeoDataFrame, nuts2_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Add NUTS2 codes to the source proxy shapefile.

        The geometries in the source shapefile will be duplicated for each NUTS2 region they intersect.
        If a source geometry intersects only with one NUTS2 polygon, it won't be considered as a fraction.
        Otherwise, the weight will be pondered based on the portion of area that fits in the NUTS2 cell.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source proxy shapefile.
        nuts2_shp : GeoDataFrame
            NUTS2 shapefile.

        Returns
        -------
        GeoDataFrame
            Source shapefile with the added NUTS2_ID column.

        Notes
        -----
        This method uses the `overlay` function to find the NUTS2 codes and add them to the source proxy shapefile.
        The resulting GeoDataFrame will have a new column ('NUTS2_ID') containing the NUTS2 codes.
        """
        if not src_shp.empty:
            src_shp = self.overlay(src_shp, nuts2_shp.reset_index(drop=False), one_intersection=True)
            src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)

        return src_shp

    def add_fid(self, src_shp: GeoDataFrame, grid_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Add FID codes to the source proxy shapefile.

        Intersects the source shapefile with the grid shapefile and duplicates each source that intersects with more
        than one FID. Weight will be pondered based on the portion of area that fits in the FID cell.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source proxy shapefile.
        grid_shp : GeoDataFrame
            Grid shapefile.

        Returns
        -------
        GeoDataFrame
            Source shapefile with the added FID code.

        Notes
        -----
        This method uses the `overlay` function to find the FID codes and add them to the source proxy shapefile.
        The resulting GeoDataFrame will have a new column ('FID') containing the FID codes.
        """
        if not src_shp.empty:
            src_shp = self.overlay(src_shp, grid_shp.reset_index(drop=False), one_intersection=True)
            src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)

        return src_shp

    def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
        """
        Sum the weights for the same FID and NUTS2 in the source proxy shapefile.

        The weight is computed based on the area that the proxy polygon occupies.

        Parameters
        ----------
        src_shp : DataFrame
            Source proxy shapefile.

        Returns
        -------
        DataFrame
            Proxy shapefile aggregated by NUTS2 and FID with the summed weights.

        Notes
        -----
        This method calculates the weight for each proxy polygon based on its area and then aggregates the weights
        for the same FID and NUTS2 using the superclass method `get_proxy_by_nut_and_fid`.
        """
        # src_shp['weight'] = src_shp.geometry.area
        if not src_shp.empty:
            src_shp = src_shp.drop(columns='geometry')
            src_shp = super().get_proxy_by_nut_and_fid(src_shp)
        else:
            src_shp = DataFrame()

        src_shp = self.broadcast(src_shp.reset_index())

        if src_shp.empty:
            return DataFrame()

        src_shp = src_shp[src_shp['FID'].isin(self.grid.get_fids().flatten())]

        src_shp = super().get_proxy_by_nut_and_fid(src_shp)

        return src_shp

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
        Perform a spatial overlay of two GeoDataFrames and merge their information.

        Parameters
        ----------
        gdf1 : GeoDataFrame
            The first GeoDataFrame containing the base geometries and associated attributes.
        gdf2 : GeoDataFrame
            The second GeoDataFrame containing geometries to overlay on the first, along with associated attributes.
        gdf1_columns : list of str, optional
            List of columns from the first GeoDataFrame to include in the resulting overlay GeoDataFrame
            (excluding geometry). If None, all columns except geometry are included.
        gdf2_columns : list of str, optional
            List of columns from the second GeoDataFrame to include in the resulting overlay GeoDataFrame
            (excluding geometry). If None, all columns except geometry are included.
        predicate : str, optional
            The spatial relationship predicate to use for the overlay operation. Default is 'intersects'.
            Other options include 'contains', 'within', etc., depending on the spatial query needs.
        one_intersection : bool, optional
            If True, geometries with only one intersection won't be divided into fractions based on area,
            and the entire geometry from gdf1 will be used. Default is False.
        weight_vars : str or list of str, optional
            Column name(s) of variables in the first GeoDataFrame that should be weighted by the proportion
            of their area that intersects with the geometries in the second GeoDataFrame. Default is None.

        Returns
        -------
        GeoDataFrame
            A new GeoDataFrame containing the result of the spatial overlay, with merged information
            from both input GeoDataFrames. The resulting GeoDataFrame has geometries and attributes
            based on the specified spatial relationship.

        Notes
        -----
        - The resulting GeoDataFrame contains information from both input GeoDataFrames based on the spatial overlay.
        - Spatial overlay is performed using the specified predicate (default is 'intersects').
        - Columns from each GeoDataFrame can be selectively included in the result.
        - If weight_vars is specified, variables are weighted by the portion of area that fits in the intersecting
          geometry.
        - This function assumes that both GeoDataFrames are in the same Coordinate Reference System (CRS).
        """

        def get_intersect_geom(x):
            """
            Determine the intersection geometry between corresponding geometries from gdf1 and gdf2.

            If `one_intersection` is True, or if there are multiple intersections, it first checks if
            geom1 is completely within geom2. If true, it returns geom1 directly; otherwise, it computes
            the intersection. In case of TopologicalError, it attempts to fix invalid geometries by buffering.

            Parameters
            ----------
            x : Series
                A row of the DataFrame containing the indices of the intersecting geometries.

            Returns
            -------
            geometry : shapely.geometry
                The intersection geometry or the original geometry from gdf1 if within geom2.
            """
            # Retrieve the geometries for the current row
            geom1 = gdf1.loc[x['gdf1_idx'], 'geometry']
            geom2 = gdf2.loc[x['gdf2_idx'], 'geometry']

            # If there are multiple intersections, or we are not concerned with fractions,
            # check if geom1 is completely within geom2
            if one_intersection or x['_num_intersect'] > 1:
                try:
                    # Perform the intersection operation
                    return geom1.intersection(geom2)
                except TopologicalError:
                    # Handle invalid geometries by buffering
                    return geom1.buffer(0).intersection(geom2.buffer(0))
            else:
                # If only one intersection, return the entire geometry of geom1
                return geom1

        def get_intersect_fraction(x):
            """
            Calculate the fraction of the area of the intersection geometry relative to the original geometry in gdf1.

            This is used to weight variables by the portion of their area that fits within the intersecting geometry.

            Parameters
            ----------
            x : Series
                A row of the DataFrame containing the intersection geometry and the index of the original geometry
                in gdf1.

            Returns
            -------
            float
                The fraction of the area of the intersection geometry relative to the original geometry in gdf1.
            """
            return x['geometry'].area / gdf1.loc[x['gdf1_idx'], 'geometry'].area

        # Initialization of columns to include
        if gdf1_columns is None:
            gdf1_columns = list(gdf1.columns)
            gdf1_columns.remove('geometry')  # Exclude geometry column
        if gdf2_columns is None:
            gdf2_columns = list(gdf2.columns)
            gdf2_columns.remove('geometry')  # Exclude geometry column
        if predicate is None:
            predicate = 'intersects'  # Default spatial relationship predicate
        if weight_vars is None:
            weight_vars = []
        elif isinstance(weight_vars, str):
            weight_vars = [weight_vars]  # Ensure weight_vars is a list

        # Ensure both GeoDataFrames are in the same CRS
        gdf1 = gdf1.reset_index(drop=True)  # Reset index of gdf1
        gdf2 = gdf2.reset_index(drop=True).to_crs(gdf1.crs)  # Reproject gdf2 to match gdf1's CRS

        # Initialize lists to hold indices of intersecting geometries
        gdf1_idx = []
        gdf2_idx = []

        # Iterate over each geometry in gdf2
        for idx, geom in enumerate(gdf2.geometry):
            # Query the spatial index of gdf1 for geometries that match the predicate with the current gdf2 geometry
            result = gdf1.sindex.query(geom, predicate=predicate)
            gdf1_idx.extend(result)  # Add the indices from gdf1 to the list
            gdf2_idx.extend([idx] * len(result))  # Add corresponding gdf2 index

        # Prepare a DataFrame to hold the results of the intersection
        intersection_df = DataFrame(columns=["gdf1_idx", "gdf2_idx"] + gdf1_columns + gdf2_columns)

        if len(gdf2_idx) > 0:
            # Populate the DataFrame with indices and attribute data from both GeoDataFrames
            intersection_df["gdf1_idx"] = array(gdf1.loc[gdf1_idx].index, dtype=uint32)
            for col_name in gdf1_columns:
                intersection_df[col_name] = gdf1.loc[gdf1_idx, col_name].values
            intersection_df["gdf2_idx"] = array(gdf2.loc[gdf2_idx].index, dtype=uint32)
            for col_name in gdf2_columns:
                intersection_df[col_name] = gdf2.loc[gdf2_idx, col_name].values

            # Calculate the intersection geometries
            if not one_intersection:
                intersection_df['_num_intersect'] = intersection_df.groupby('gdf1_idx')['gdf1_idx'].transform('size')
            intersection_df['geometry'] = intersection_df.apply(get_intersect_geom, axis=1)
            if not one_intersection:
                intersection_df.drop(columns='_num_intersect', inplace=True)

            # Apply area-based weighting to specified variables, if any
            if len(weight_vars) > 0:
                intersection_df['_fraction'] = array(intersection_df.apply(get_intersect_fraction, axis=1),
                                                     dtype=precision)

                for var_name in weight_vars:
                    # Apply weighting to each specified variable
                    intersection_df[var_name] = array(
                        intersection_df[var_name], dtype=precision) * intersection_df['_fraction']
                intersection_df.drop(columns='_fraction', inplace=True)

        else:
            # If no intersections, set geometry column to None
            intersection_df['geometry'] = None

        # Convert the resulting DataFrame to a GeoDataFrame with the appropriate CRS
        intersection_df = GeoDataFrame(intersection_df, crs=gdf1.crs)
        return intersection_df

    # # TODO: It is not used. Remove it?
    # def get_proxy_by_nut_and_fid_current(self, src_shp: GeoDataFrame) -> DataFrame:
    #     """
    #     Sum the weights for the same FID and NUTS2 in the source proxy shapefile.
    #
    #     The weight is computed based on the area that the proxy polygon occupies.
    #
    #     Parameters
    #     ----------
    #     src_shp : DataFrame
    #         Source proxy shapefile.
    #
    #     Returns
    #     -------
    #     DataFrame
    #         Proxy shapefile aggregated by NUTS2 and FID with the summed weights.
    #
    #     Notes
    #     -----
    #     This method calculates the weight for each proxy polygon based on its area and then aggregates the weights
    #     for the same FID and NUTS2 using the superclass method `get_proxy_by_nut_and_fid`.
    #     """
    #     # src_shp['weight'] = src_shp.geometry.area
    #     src_shp = src_shp.drop(columns='geometry')
    #
    #     src_shp = super().get_proxy_by_nut_and_fid(src_shp)
    #
    #     return src_shp

    # def add_info_to_src_shp_current(self, src_shp: GeoDataFrame) -> GeoDataFrame:
    #     """
    #     Add the NUTS2_codes and FID to the src_shp.
    #
    #     Parameters
    #     ----------
    #     src_shp : GeoDataFrame
    #         Source proxy shapefile.
    #
    #     Returns
    #     -------
    #     GeoDataFrame
    #         Source shapefile with the NUTS2_ID and FID columns.
    #
    #     Notes
    #     -----
    #     - This method adds NUTS2 codes using the add_nut_codes method.
    #     - It also adds destination cell FID using the add_fid method.
    #     """
    #     # print(f"Rank: {self.rank}, Raster: {len(src_shp)} ")
    #     log_message("Adding NUTS_code.", level=3)
    #     src_shp = self.add_nut_codes(src_shp, self.nuts2_shp)
    #     # print(f"Rank: {self.rank}, NUTS: {len(src_shp)} ")
    #
    #     src_shp = self.broadcast(src_shp)
    #
    #     log_message("Adding FIDs.", level=3)
    #     src_shp = self.add_fid(src_shp, self.grid.shapefile)
    #     # print(f"Rank: {self.rank}, FID: {len(src_shp)} ")
    #
    #     # Using the area as weight
    #     src_shp['weight'] = src_shp.geometry.area
    #
    #     return src_shp

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
        log_message("Adding NUTS_code.", level=3)

        if not src_shp.empty:
            # Get the bounding box of the in-memory GeoDataFrame
            min_x, min_y, max_x, max_y = src_shp.total_bounds

            # Create a bounding box tuple
            bbox = (min_x, min_y, max_x, max_y)

            nuts2_shp = read_file(self.nuts2_shp_path, bbox=bbox)
            nuts2_shp.set_index(self.nuts2_shp_name, inplace=True)

            src_shp = self.add_nut_codes(src_shp, self.nuts2_shp)

            if not src_shp.empty:
                log_message("Adding FIDs.", level=3)

                grid_shp = read_file(self.grid.global_attrs["aux_shapefile"], bbox=bbox)
                grid_shp.set_index('FID', inplace=True)

                src_shp = self.add_fid(src_shp, grid_shp)

                # Using the area as weight
                src_shp['weight'] = src_shp.geometry.area

        return src_shp
