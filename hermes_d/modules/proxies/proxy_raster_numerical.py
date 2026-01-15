from hermes_d.modules.proxies import ProxyRaster
from rasterio import open as open_raster
from rasterio.windows import from_bounds
from rasterio.features import shapes
from numpy import array, float64, float32, random, uint32, floor
from shapely.geometry import shape
from geopandas import GeoDataFrame
from nes import Nes
from gc import collect
from typing import Union, List, Optional
from pandas import DataFrame


class ProxyRasterNum(ProxyRaster):
    """
    Represents a numerical Proxy for HERMESv3_Delta, extending the functionality of ProxyRaster.

    Attributes
    ----------
    (Inherits attributes from ProxyRaster)

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
                 src_path: str, nuts2_list: List[str]):
        """
        Initialize the ProxyRasterNum class.

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
        nuts2_list : List[str]
            List of NUTS2 codes to be used for the analysis.

        Notes
        -----
        This class extends the functionality of ProxyRaster and is designed specifically for handling numerical proxy
        data.
        """
        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list)

    def read_src_proxy(self, path: str, clip: Optional[GeoDataFrame] = None) -> GeoDataFrame:
        """
        Read the source proxy raster file and convert it into a GeoDataFrame.

        Parameters
        ----------
        path : str
            Path to the source proxy raster file.
        clip : GeoDataFrame, optional
            Region of the raster to read. If not provided, the entire raster is considered.

        Returns
        -------
        GeoDataFrame
            Source proxy shapefile containing proxy weights based on the raster data.

        Notes
        -----
        This method reads the source proxy raster file, converts it into a GeoDataFrame, and assigns weights based on
        the raster data. The resulting GeoDataFrame includes geometry and weight columns.

        """
        dst_crs = self.grid.shapefile.crs

        if clip is None:
            clip = self.clip

        if len(clip) > 0:
            with open_raster(path) as src:
                # Creating windows to read a portion
                bbox = array(clip.to_crs(src.crs).total_bounds, dtype=float64)
                my_window = self.fix_window(from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform))
                if my_window is not None:
                    # Reading data and mask
                    data = src.read(1, window=my_window, masked=True)

                    # Parsing the data
                    data = floor(data).astype(float32)  # Get the integer part
                    data += random.uniform(size=data.shape).astype(float32)  # Add a random decimal part
                    data[data < 1] = 0  # Keep old zeros as zeros
                    my_mask = data > 0
                    if my_mask.any():
                        # Use a generator instead of a list
                        shape_gen = ((shape(s), v)
                                     for s, v in shapes(data, mask=my_mask, transform=src.window_transform(my_window)))
                        # or build a dict from unpacked shapes
                        proxy_shp = GeoDataFrame(dict(zip(["geometry", "weight"], zip(*shape_gen))), crs=src.crs)
                        collect()
                        # Weight as integer and filtering

                        proxy_shp = proxy_shp.loc[proxy_shp["weight"] >= 1]
                        proxy_shp["weight"] = floor(proxy_shp["weight"]).astype(uint32)

                        # Reproject data
                        proxy_shp = proxy_shp.to_crs(dst_crs)

                        collect()
                    else:
                        proxy_shp = GeoDataFrame()
                else:
                    proxy_shp = GeoDataFrame()
        else:
            proxy_shp = GeoDataFrame()

        return proxy_shp

    def add_nut_codes(self, src_shp: GeoDataFrame, nuts2_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the NUTS2 codes and add them to a new column in the source shapefile.

        The geometries will be duplicated as many times as they appear in different NUTS2 regions.
        If the source geometry intersects only with one NUTS2 polygon, it won't be considered as a fraction.
        Otherwise, the weight will be pondered with the portion of the area that fits in the FID cell.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source shapefile.
        nuts2_shp : GeoDataFrame
            NUTS2 shapefile.

        Returns
        -------
        GeoDataFrame
            Source shapefile with the NUT2_ID code.

        Notes
        -----
        This method performs an overlay operation between the source shapefile (`src_shp`) and the NUTS2 shapefile
        (`nuts2_shp`). The resulting GeoDataFrame includes a new column with NUT2_ID codes.

        """
        if not src_shp.empty:
            src_shp = self.overlay(src_shp, nuts2_shp.reset_index(drop=False), one_intersection=False,
                                   weight_vars='weight')
            src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)
        return src_shp

    def add_fid(self, src_shp: GeoDataFrame, grid_shp: GeoDataFrame) -> GeoDataFrame:
        """
        Find the FID codes and add them to a new column in the source shapefile.

        Intersects the source shapefile (`src_shp`) with the grid shapefile (`grid_shp`) and duplicates each source
        that intersects with more than one FID. The weight is pondered with the portion of the area that fits in the
        FID cell.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source shapefile.
        grid_shp : GeoDataFrame
            Grid shapefile.

        Returns
        -------
        GeoDataFrame
            Source shapefile with the FID code.

        Notes
        -----
        This method performs an overlay operation between the source shapefile and the grid shapefile, resulting in a
        GeoDataFrame with a new column containing FID codes.

        """
        if not src_shp.empty:
            src_shp = self.overlay(src_shp, grid_shp.reset_index(drop=False), one_intersection=True,
                                   weight_vars='weight')
            src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)
        return src_shp

    def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
        """
        Sum the weights that go to the same FID and NUTS2.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source proxy shapefile

        Returns
        -------
        DataFrame
            Proxy shapefile aggregated by NUTS2 and FID with the summed weights.
        """
        if not src_shp.empty:
            src_shp = src_shp.drop(columns='geometry')
            src_shp = super().get_proxy_by_nut_and_fid(src_shp)
        else:
            src_shp = DataFrame()

        return src_shp
