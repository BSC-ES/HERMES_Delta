from hermes_d.modules.proxies import ProxyRasterNum
from rasterio import open as open_raster
from rasterio.windows import from_bounds
from rasterio.features import shapes
from shapely.geometry import shape
from numpy import array, isin, floor, random, uint32, float32, float64
from geopandas import GeoDataFrame
from nes import Nes
from gc import collect
from typing import Union, List, Optional


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

    def read_src_proxy(self, path: str, clip: Optional[GeoDataFrame] = None) -> GeoDataFrame:
        """
        Read the source proxy raster file and convert it into a GeoDataFrame.

        The raster will be masked with the corresponding categories of the categorical proxy data.

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

                    # Reading categorized raster
                    with open_raster(self.category_path) as src_cat:
                        cat_mask = src_cat.read(1, window=my_window, masked=True)
                        if data.shape != cat_mask.shape:
                            msg = f"Numerical raster {path} and categorical raster {self.category_path} are different"
                            raise RuntimeError(msg)
                        # Creating mask with selected categories
                        cat_mask[~isin(cat_mask, self.category_list)] = 0
                        cat_mask[isin(cat_mask, self.category_list)] = 1

                    # Making the numerical values for the places where the category is
                    data *= cat_mask
                    del cat_mask

                    # Parsing the data
                    data = floor(data).astype(float32)  # Get the integer part
                    data += random.uniform(size=data.shape).astype(float32)  # Add a random decimal part
                    data[data < 1] = 0  # Keep old zeros as zeros
                    my_mask = data > 0

                    # Use a generator instead of a list
                    shape_gen = ((shape(s), v) for s, v in shapes(
                        data, mask=my_mask, transform=src.window_transform(my_window)))

                    # # or build a dict from unpacked shapes
                    # # proxy_shp = GeoDataFrame(dict(zip(["geometry", "weight"], zip(*shape_gen))), crs=src.crs)
                    # # Unpack the shapes and weights
                    # geometries, weights = zip(*shape_gen)
                    #
                    # # Create the GeoDataFrame
                    # proxy_shp = GeoDataFrame({'geometry': geometries, 'weight': weights}, crs=src.crs)
                    #
                    # collect()

                    # Convert generator to list for debugging purposes
                    shape_list = list(shape_gen)
                    if not shape_list:
                        proxy_shp = GeoDataFrame(columns=['weight'], geometry=[], crs=dst_crs)
                    else:
                        # Unpack the shapes and weights
                        geometries, weights = zip(*shape_list)

                        # Create the GeoDataFrame
                        proxy_shp = GeoDataFrame({'geometry': geometries, 'weight': weights}, crs=src.crs)

                        collect()
                        # Weight as integer and filtering
                        proxy_shp = proxy_shp.loc[proxy_shp["weight"] >= 1]
                        proxy_shp["weight"] = floor(proxy_shp["weight"]).astype(uint32)
                    collect()

                    # Reproject data
                    proxy_shp = proxy_shp.to_crs(dst_crs)
                else:
                    proxy_shp = GeoDataFrame()
        else:
            # Empty proxy:
            proxy_shp = GeoDataFrame()

        return proxy_shp
