from hermes_d.modules.proxies import Proxy
from nes import Nes
from numpy import array_split, isnan
from pandas import DataFrame, concat
from geopandas import GeoDataFrame, read_file
from rasterio.windows import Window
from math import modf
from typing import Union, List


class ProxyRaster(Proxy):
    """
    Represents a ProxyRaster for HERMESv3_Delta, extending the functionality of the Proxy class.

    Attributes
    ----------
    Inherits attributes from the Proxy class.

    Notes
    -----
    The ProxyRaster class inherits attributes and methods from the Proxy class.

    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str], src_path: str,
                 nuts2_list: List[str], clip_nuts2=False):
        """
        Initialize the ProxyRaster class.

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
        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list, clip_nuts2=clip_nuts2)

    @staticmethod
    def fix_window(window):
        """
        Adjusts the window parameters to ensure entire cells are selected by rounding
        the offsets and adjusting the width and height accordingly.

        Parameters
        ----------
        window : Window
            The input window with potentially fractional offsets and dimensions.

        Returns
        -------
        Window or None
            A new Window object with integer offsets and dimensions adjusted to
            cover entire cells, or None if any of the window components is None or NaN.
        """
        # Check if any component of the window is None or NaN
        if (window.col_off is None or isnan(window.col_off) or
                window.row_off is None or isnan(window.row_off) or
                window.width is None or isnan(window.width) or
                window.height is None or isnan(window.height)):
            return None
        # Correcting indices to get entire cells
        col_off = int(window.col_off)  # Get first entire col
        row_off = int(window.row_off)  # Get first entire row
        # math.modf returns a tuple with the decimal part as the first element
        width = int(window.width + modf(window.col_off)[0]) + 1
        height = int(window.height + modf(window.row_off)[0]) + 1

        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

        return window
