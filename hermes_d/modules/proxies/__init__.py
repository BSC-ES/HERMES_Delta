from .proxy_default import Proxy
from .proxy_raster import ProxyRaster
from .proxy_raster_numerical import ProxyRasterNum
from .proxy_raster_categorical import ProxyRasterCat
from .proxy_shapefile import ProxyShp
from .proxy_shapefile_point import ProxyShpPoint
from .proxy_shapefile_polygon import ProxyShpPoly
from .proxy_shapefile_line import ProxyShpLine
from .proxy_raster_numerical_categorized import ProxyRasterNumCat
from .double_proxy import DoubleProxy

__all__ = ["Proxy", "ProxyRaster", "ProxyRasterNum", "ProxyRasterCat", "ProxyShp", "ProxyShpPoint", "ProxyShpPoly",
           "ProxyShpLine", "ProxyRasterNumCat", "DoubleProxy"]
