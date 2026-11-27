from hermes_d.modules.proxies import ProxyRasterNum
from rasterio import open as open_raster
from numpy import array, isin
from numpy.ma import getmaskarray, masked_array
from geopandas import GeoDataFrame
from nes import Nes
from typing import Union, List


class ProxyRasterNumCat(ProxyRasterNum):
    """
    Represents a numerical-categorized Proxy for HERMESv3_Delta, extending the functionality of ProxyRasterNum.

    Attributes
    ----------
    (Inherits attributes from ProxyRasterNum)
    category_path : str
        Path to the source file containing the categorical proxy data.
    category_list : List[int]
        List of categories to filter the numerical proxy data.

    Methods
    -------
    __init__(grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str],
             src_path: str, nuts2_list: List[str])
        Initialize the ProxyRasterNum class.

    Notes
    -----
    ProxyRasterNum is a subclass of ProxyRaster, specifically designed for handling numerical proxy data.

    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str],
                 src_path: str, category_path: str, category_list: List[int], nuts2_list: List[str]):
        """
        Initialize the ProxyRasterNumCat class.

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
            Path to the source file containing the numerical proxy data.
        category_path : str
            Path to the source file containing the categorical proxy data.
        category_list : List[int]
            List of categories to filter the numerical proxy data from.
        nuts2_list : List[str]
            List of NUTS2 codes to be used for the analysis.

        Notes
        -----
        This class extends the functionality of ProxyRaster and is designed specifically for handling numerical proxy
        data.
        """
        if category_list is None or len(category_list) == 0:
            raise ValueError("Unable to create a categorized proxy. No set categories found.")
        self.category_path = category_path
        self.category_list = array(category_list)

        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list)

    def read_window_data(self, src, window):
        """Read numerical pixels selected by an aligned categorical raster.

        Parameters
        ----------
        src : rasterio.io.DatasetReader
            Open numerical raster.
        window : rasterio.windows.Window
            Nonempty window owned by this rank.

        Returns
        -------
        numpy.ma.MaskedArray
            Numerical values masked by nodata and the selected categories.

        Raises
        ------
        RuntimeError
            If numerical and categorical rasters do not share a pixel grid.
        """
        data = super().read_window_data(src, window)
        with open_raster(self.category_path) as categories:
            if (src.crs != categories.crs or src.transform != categories.transform or
                    src.width != categories.width or src.height != categories.height):
                raise RuntimeError(f"Numerical raster {src.name} and categorical raster "
                                   f"{self.category_path} must have identical CRS, transform and dimensions")
            category_data = super().read_window_data(categories, window)
        selected = isin(category_data.data, self.category_list) & ~getmaskarray(category_data)
        return masked_array(data, mask=getmaskarray(data) | ~selected)
