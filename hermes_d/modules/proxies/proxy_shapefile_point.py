from hermes_d.modules.proxies import ProxyShp
from geopandas import GeoDataFrame
from typing import List, Union
from nes import Nes


class ProxyShpPoint(ProxyShp):
    """
    ProxyShpPoint represents a point-based proxy for HERMESv3_Delta.

    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str],
                 src_path: str, nuts2_list: List[str], shp_column_name: str, shp_column_value: str):
        """
        Initialize the ProxyShpPoint class.

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

    def add_nut_codes(self, src_shp: GeoDataFrame, nuts2_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the NUTS2 codes and put them in a new column.

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
        return super().add_nut_codes(src_shp, nuts2_shp)

    def add_fid(self, src_shp: GeoDataFrame, grid_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the FID codes and put them in a new column.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Sources shapefile
        grid_shp : GeoDataFrame
            Grid shapefile

        Returns
        -------
        GeoDataFrame
            Sources shapefile with the FID
        """
        return super().add_fid(src_shp, grid_shp)

    # def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
    #     """
    #     Sum the weights that go to the same FID and NUTS2.
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
    #     src_shp = src_shp.drop(columns='geometry')
    #     src_shp = super().get_proxy_by_nut_and_fid(src_shp)
    #     return src_shp
