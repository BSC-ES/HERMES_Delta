from hermes_d.modules.proxies import ProxyShp
from geopandas import GeoDataFrame
from nes import Nes
from typing import Union, List


class ProxyShpPoly(ProxyShp):
    """
    ProxyShpPoly represents a polygon-based proxy for HERMESv3_Delta.

    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str], src_path: str,
                 nuts2_list: List[str], shp_column_name: str, shp_column_value: str):
        """
        Initialize the ProxyShpPoly class.

        Parameters
        ----------
        grid : Nes
            Grid
        clip : GeoDataFrame
            Clip
        nuts2_shp : GeoDataFrame or str
            NUTS2 shapefile, or path to it
        src_path : str
            Path to the source file containing the proxy data
        nuts2_list : list of str
            List of NUTS2 codes to use
        shp_column_name : str
            Column of the shapefile to use as a proxy value
        shp_column_value : str
            Proxy value to filter in the shp_column_name
        """
        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list, shp_column_name, shp_column_value)

    def add_nut_codes(self, src_shp: GeoDataFrame, nuts2_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the NUTS2 codes and put them in a new column.

        The geometries will be duplicated as many times as it appears in different NUTS2 regions.
        If the source geometry intersects only with one NUTS2 polygon, it won't be considered as a fraction.
        If not, weight will be pondered with the portion of the area that fits in the FID cell.

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
        src_shp = self.overlay(src_shp, nuts2_shp.reset_index(drop=False), one_intersection=False,
                               weight_vars='weight')
        src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)

        return src_shp

    def add_fid(self, src_shp: GeoDataFrame, grid_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the FID codes and put them in a new column.

        Intersects the sources shapefile with the grid shapefile and duplicates each source that intersects with more
        than one FID.
        Weight will be pondered with the portion of area that fits in the FID cell.

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
        src_shp = self.overlay(src_shp, grid_shp.reset_index(drop=False), one_intersection=True,
                               weight_vars='weight')
        src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)

        return src_shp

    # def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
    #     """
    #     Sum the weights that go to the same FID and NUTS2.
    #     Weight is pondered by their area.
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
    #
    #     src_shp = src_shp.drop(columns='geometry')
    #     src_shp = super().get_proxy_by_nut_and_fid(src_shp)
    #     return src_shp
