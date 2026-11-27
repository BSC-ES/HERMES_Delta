from hermes_d.utilities import get_raster_area_fractions, get_raster_polygon_areas
from hermes_d.modules.proxies import ProxyRaster
from hermes_d.utilities import redistribute_dataframes as redistribute
from rasterio.features import shapes
from rasterio import open as open_raster
from numpy import isin, array, where, uint16, uint32
from shapely.errors import TopologicalError
from geopandas import GeoDataFrame
from pandas import DataFrame
from nes import Nes
from typing import Union, List, Optional
from hermes_d.config import precision


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
            Retained for compatibility. Reading uses the global domain window.

        Returns
        -------
        GeoDataFrame
            Geometries in the native raster CRS representing the source proxy.
        No reprojection or resampling of the raster is performed.

        """
        window = self.get_rank_window(path)

        with open_raster(path) as src:
            if window.width == 0 or window.height == 0:
                gdf = GeoDataFrame({'weight': []}, geometry=[], crs=src.crs)
            else:
                gdf = self.polygonize_categories(src, window)
        gdf = redistribute(gdf, self.comm)
        self.comm.Barrier()
        return gdf

    def polygonize_categories(self, src, window):
        """Polygonize valid selected categories in one nonempty raster window.

        Parameters
        ----------
        src : rasterio.io.DatasetReader
            Open categorical raster.
        window : rasterio.windows.Window
            Nonempty window owned by this rank.

        Returns
        -------
        GeoDataFrame
            Selected polygons in the native raster CRS.
        """
        data = self.read_window_data(src, window)

        # Apply mask: only keep allowed values and categorize as 1
        mask = isin(data.data, self.category_list) & ~data.mask
        data = where(mask, 1, 0)

        data = data.astype(uint16)

        transform = src.window_transform(window)
        if mask.any():
            shapes_gen = shapes(data, mask=mask, transform=transform)
            gdf = GeoDataFrame.from_features(
                ({'geometry': shape, 'properties': {'weight': value}} for shape, value in shapes_gen),
                crs=src.crs
            )
        else:
            gdf = GeoDataFrame({'weight': []}, geometry=[], crs=src.crs)
        return gdf

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
        - Area fractions use the native raster CRS: planar for projected rasters
          and geodesic for geographic rasters.
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
                intersection_df['_fraction'] = array(get_raster_area_fractions(
                    gdf1, GeoDataFrame(intersection_df, crs=gdf1.crs)), dtype=precision)

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
        """Intersect source strips and assign fragment areas as categorical weights.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Selected categorical polygons in the native raster CRS.

        Returns
        -------
        GeoDataFrame
            Fragments weighted by their area in square metres.
        """
        src_shp = super().add_info_to_src_shp(src_shp)
        if not src_shp.empty:
            src_shp['weight'] = get_raster_polygon_areas(src_shp)
        return src_shp
