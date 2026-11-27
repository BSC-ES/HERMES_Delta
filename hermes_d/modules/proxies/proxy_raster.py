from hermes_d.modules.proxies import Proxy
from nes import Nes
from numpy import isfinite
from pandas import DataFrame
from geopandas import GeoDataFrame, read_file
from rasterio.windows import Window, from_bounds
from rasterio import open as open_raster
from rasterio.warp import transform_bounds
from math import floor, ceil
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
        """Round a window outwards and add one precautionary column and row.

        Parameters
        ----------
        window : Window
            Window with possibly fractional offsets and dimensions.

        Returns
        -------
        Window or None
            Enclosing integer window with one extra column and row at the upper
            pixel-index limits, or None for invalid bounds. The global reader
            clips this margin to the raster extent before MPI partitioning.
        """
        if window is None or not all(isfinite(value) for value in window.flatten()):
            return None
        left, top = floor(window.col_off), floor(window.row_off)
        right = ceil(window.col_off + window.width)
        bottom = ceil(window.row_off + window.height)
        return Window(left, top, right - left + 1, bottom - top + 1)

    def get_total_window(self, path: str) -> Window:
        """Calculate and broadcast the global window clipped to the raster extent.

        Parameters
        ----------
        path : str
            Source raster path.

        Returns
        -------
        Window
            Integer window shared by all ranks, possibly empty.
        """
        bbox = self.get_bbox(full_domain=True)
        window = None
        if self.master:
            with open_raster(path) as src:
                bounds = bbox.to_crs(src.crs).total_bounds
                if all(isfinite(bounds)):
                    window = self.fix_window(from_bounds(*bounds, transform=src.transform))
                if window is None:
                    window = Window(0, 0, 0, 0)
                else:
                    left = min(src.width, max(0, int(window.col_off)))
                    top = min(src.height, max(0, int(window.row_off)))
                    right = min(src.width, max(0, int(window.col_off + window.width)))
                    bottom = min(src.height, max(0, int(window.row_off + window.height)))
                    window = Window(left, top, max(0, right - left), max(0, bottom - top))
        return self.comm.bcast(window)

    def get_partial_window_x(self, total_window: Window) -> Window:
        """Assign a disjoint strip of complete raster columns to this rank.

        Parameters
        ----------
        total_window : Window
            Global integer window.

        Returns
        -------
        Window
            Rank window. Ranks beyond the available columns receive zero width.
        """
        columns, extra = divmod(int(total_window.width), self.size)
        start = int(total_window.col_off) + self.rank * columns + min(self.rank, extra)
        width = columns + int(self.rank < extra)
        return Window(start, total_window.row_off, width, total_window.height)

    def get_partial_window(self, total_window: Window) -> Window:
        """Partition a global window into column strips.

        Parameters
        ----------
        total_window : Window
            Global integer window.

        Returns
        -------
        Window
            Window owned by this rank.
        """
        return self.get_partial_window_x(total_window)

    def get_rank_window(self, path: str) -> Window:
        """Obtain this rank's part of the global raster window.

        Parameters
        ----------
        path : str
            Source raster path.

        Returns
        -------
        Window
            Window owned by this rank.
        """
        return self.get_partial_window(self.get_total_window(path))

    def read_window_data(self, src, window: Window):
        """Read one raster band with its nodata mask.

        Parameters
        ----------
        src : rasterio.io.DatasetReader
            Open source raster.
        window : Window
            Nonempty integer window owned by this rank.

        Returns
        -------
        numpy.ma.MaskedArray
            Band values with invalid pixels masked.
        """
        return src.read(1, window=window, masked=True)

    def add_info_to_src_shp(self, src_shp: GeoDataFrame) -> GeoDataFrame:
        """Intersect source strips with NUTS and all overlapping destination cells.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Source polygons in the native raster CRS.

        Returns
        -------
        GeoDataFrame
            Fragments with NUTS and FID identifiers, including remote FIDs.

        Notes
        -----
        Raster ownership is independent of destination ownership. Read candidates
        from the global auxiliary grid rather than the rank's local grid.
        """
        src_shp = self.add_nut_codes(src_shp, self.nuts2_shp)
        if not src_shp.empty:
            bbox = transform_bounds(src_shp.crs, self.grid.shapefile.crs, *src_shp.total_bounds)
            grid_shp = read_file(self.grid.global_attrs['aux_shapefile'], bbox=bbox).set_index('FID')
            src_shp = self.add_fid(src_shp, grid_shp)
        return src_shp

    def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
        """Aggregate source fragments and deliver weights to destination owners.

        Parameters
        ----------
        src_shp : GeoDataFrame
            Weighted fragments with NUTS and FID identifiers.

        Returns
        -------
        DataFrame
            Weights grouped by NUTS and FID for this rank's destination cells.

        Notes
        -----
        Every rank participates even when its raster strip is empty. Only the
        grouped weights are gathered and broadcast, not individual geometries.
        """
        if not src_shp.empty:
            local = super().get_proxy_by_nut_and_fid(src_shp.drop(columns='geometry'))
        else:
            local = DataFrame()
        weights = self.broadcast(local.reset_index())
        if weights.empty:
            return DataFrame()
        weights = weights[weights['FID'].isin(self.grid.get_fids().flatten())]
        return super().get_proxy_by_nut_and_fid(weights)
