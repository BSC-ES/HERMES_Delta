from hermes_d.modules.proxies import ProxyRaster
from rasterio import open as open_raster
from rasterio.features import shapes
from numpy import array, uint32, floor, arange, int32, iinfo, isfinite
from numpy.ma import getmaskarray, getdata
from shapely.geometry import shape
from geopandas import GeoDataFrame
from nes import Nes
from gc import collect
from typing import Union, List, Optional
from hermes_d.utilities import get_raster_area_fractions


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
            Retained for compatibility. Reading uses the global domain window.

        Returns
        -------
        GeoDataFrame
            Source proxy shapefile containing proxy weights based on the raster data.

        Notes
        -----
        This method reads the source proxy raster file, converts it into a GeoDataFrame, and assigns weights based on
        the raster data. Each valid pixel remains a separate polygon with its
        original truncated weight; no random offsets or float32 conversion are
        applied to distinguish neighbouring pixels.

        Polygon geometries retain the native numerical raster CRS. NUTS and
        destination cells are transformed to this CRS by the overlay. Projected
        raster area fractions are planar; geographic rasters retain geodesic
        measurement.

        """
        window = self.get_rank_window(path)
        with open_raster(path) as src:
            if window.width == 0 or window.height == 0:
                return GeoDataFrame({'weight': []}, geometry=[], crs=src.crs)
            data = self.read_window_data(src, window)
            proxy_shp = self.polygonize_pixels(data, src.window_transform(window), src.crs)
        collect()
        return proxy_shp

    @staticmethod
    def polygonize_pixels(data, transform, crs) -> GeoDataFrame:
        """Create one polygon per valid numerical raster pixel deterministically.

        Parameters
        ----------
        data : numpy.ndarray or numpy.ma.MaskedArray
            Two-dimensional numerical raster values. Masked, non-finite and
            values below one after truncation are excluded.
        transform : affine.Affine
            Transformation from pixel coordinates to the source CRS, including
            the offset of the raster window being read.
        crs : rasterio.crs.CRS or str
            Coordinate reference system of the raster geometries.

        Returns
        -------
        GeoDataFrame
            Separate pixel polygons with a ``weight`` column containing the
            original values truncated with ``floor`` and cast to ``uint32``.
            Empty inputs or inputs without valid pixels return an empty frame
            with the source CRS and the same columns.

        Raises
        ------
        ValueError
            If the raster window exceeds the supported number of int32 labels.

        Notes
        -----
        Polygonization operates on unique integer pixel labels, not on the
        numerical proxy values. Weights are retrieved from the original pixels,
        preserving the existing integer-weight convention. Equal neighbouring
        weights cannot merge, and no random decimal or float32 conversion is
        applied to the weights. Input values are not modified.
        """
        values = getdata(floor(data))
        mask = ~getmaskarray(data) & isfinite(values) & (values >= 1)
        if not mask.any():
            return GeoDataFrame({'weight': array([], dtype=uint32)}, geometry=[], crs=crs)
        if values.size > iinfo(int32).max:
            raise ValueError('Raster window is too large for int32 pixel labels; use a smaller window.')

        labels = arange(values.size, dtype=int32).reshape(values.shape)
        shape_gen = ((shape(geometry), values.flat[int(label)])
                     for geometry, label in shapes(labels, mask=mask, transform=transform))
        result = GeoDataFrame(dict(zip(['geometry', 'weight'], zip(*shape_gen))), crs=crs)
        result['weight'] = array(result['weight'], dtype=uint32)
        return result

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
                                   weight_vars='weight', fraction_calculator=get_raster_area_fractions)
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
                                   weight_vars='weight', fraction_calculator=get_raster_area_fractions)
            src_shp.drop(columns=["gdf1_idx", "gdf2_idx"], inplace=True)
        return src_shp
