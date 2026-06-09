from hermes_d.modules.proxies import ProxyShp
from geopandas import GeoDataFrame
from pandas import DataFrame
from numpy import uint32, array
from typing import Union, List, Optional
from hermes_d.config import precision
from nes import Nes


class ProxyShpLine(ProxyShp):
    """
    ProxyShpLine represents a line-based proxy for HERMESv3_Delta.

    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str],
                 src_path: str, nuts2_list: List[str], shp_column_name: str, shp_column_value: str):
        """
        Initialize the ProxyShpLine class.

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
        shp_column_name : str
            Column of the shapefile to use as a proxy value.
        shp_column_value: str
            Proxy value to filter in the shp_column_name.
        """
        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list, shp_column_name, shp_column_value)

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
        src_shp = super().add_info_to_src_shp(src_shp)

        src_shp['weight'] *= src_shp.geometry.length

        return src_shp

    def add_nut_codes(self, src_shp: GeoDataFrame, nuts2_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the NUTS2 codes and put them in a new column.

        The geometries will be duplicated as many times as it appears in different NUTS2 regions.
        If the source geometry intersects only with one NUTS2 polygon, it won't be considered as a fraction.
        If not, the weight will be pondered with the portion of length that fits in the FID cell.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Sources shapefile
        nuts2_shp : GeoDataFrame
            NUTS2 shapefile

        Returns
        -------
        GeoDataFrame
            Sources shapefile with the NUT2_ID code
        """
        src_shp = self.overlay(src_shp, nuts2_shp.reset_index(drop=False), one_intersection=False)
        src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)

        return src_shp

    def add_fid(self, src_shp: GeoDataFrame, grid_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the FID codes and put them in a new column.

        Intersects the sources shapefile with the grid shapefile and duplicates each source that intersects with more
        than one FID.
        Weight will be pondered with the portion of length that fits in the FID cell.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Sources shapefile
        grid_shp : GeoDataFrame
            Grid shapefile

        Returns
        -------
        GeoDataFrame
            Sources shapefile with the FID code
        """
        src_shp = self.overlay(src_shp, grid_shp.reset_index(drop=False), one_intersection=True)
        src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)

        return src_shp

    # def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
    #     """
    #     Sum the weights that go to the same FID and NUTS2.
    #     Weight is pondered by their length.
    #
    #     Parameters
    #     ----------
    #     src_shp : GeoDataFrame
    #         Source proxy shapefile
    #
    #     Returns
    #     -------
    #     DataFrame
    #         Proxy shapefile aggregated by NUTS2 and FID with the summed weights.
    #     """
    #     if not src_shp.empty:
    #         src_shp = src_shp.drop(columns='geometry')
    #         src_shp = super().get_proxy_by_nut_and_fid(src_shp)
    #     else:
    #         src_shp = DataFrame()
    #     return src_shp

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
                geom = gdf1.loc[x['gdf1_idx'], 'geometry'].intersection(
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
