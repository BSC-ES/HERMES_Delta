from hermes_d.modules.proxies import Proxy
from pandas import concat, DataFrame
from geopandas import GeoDataFrame, read_file
from numpy import array
from hermes_d.config import precision
from nes import Nes
from mpi4py.MPI import Comm
from os import path, makedirs
from typing import List, Union, Optional


class ProxyShp(Proxy):
    """
    Represents a Proxy for HERMESv3_Delta with additional filtering based on shapefile properties.

    Attributes
    ----------
    shp_column_name : str
        Column of the shapefile to use as proxy value.
    shp_column_value: str
        Proxy value to filter in the shp_column_name.
    """
    def __init__(self, grid: Nes, clip: GeoDataFrame, nuts2_shp: Union[GeoDataFrame, str],
                 src_path: str, nuts2_list: List[str], shp_column_name: str, shp_column_value: str):
        """
        Initialize the ProxyShp class.

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
            Column of the shapefile to use as proxy value.
        shp_column_value: str
            Proxy value to filter in the shp_column_name.
        """
        self.shp_column_name = shp_column_name
        self.shp_column_value = shp_column_value
        super().__init__(grid, clip, nuts2_shp, src_path, nuts2_list)

    def read_src_proxy(self, filepath: str, clip: Optional[GeoDataFrame] = None) -> GeoDataFrame:
        """
        Read the source proxy and return it as a shapefile.

        Parameters
        ----------
        filepath : str
            Path to the source proxy file.
        clip : GeoDataFrame, optional
            Region of the raster to read.

        Returns
        -------
        GeoDataFrame
            Source proxy shapefile.
        """
        if clip is None:
            clip = self.clip

        if len(clip) > 0:
            # Read and filter shapefile based on provided clip region
            proxy_shp = read_file(filepath, bbox=clip).to_crs(clip.crs)
            proxy_shp['weight'] = array(proxy_shp['weight'], dtype=precision)
            proxy_shp = proxy_shp.loc[proxy_shp[self.shp_column_name] == self.shp_column_value, ['weight', 'geometry']]
        else:
            # Empty proxy if clip is not provided
            proxy_shp = GeoDataFrame(columns=['weight'], geometry=[], crs=self.clip.crs)

        return proxy_shp

    def get_proxy_by_nut_and_fid(self, src_shp: GeoDataFrame) -> DataFrame:
        """
        Sum the weights that go to the same FID and NUTS2.
        Weight is pondered by their length.

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


def write_shapefile_parallel(comm: Comm, data: GeoDataFrame, filepath: str) -> bool:
    """
    Write a scattered shapefile.

    Parameters
    ----------
    comm : Comm
        MPI communicator
    data : GeoDataFrame
        Portion of the shapefile
    filepath : str
        Output path

    Returns
    -------
    bool
        True if the write operation is successful.
    """
    # Gather data from all processes to the root (rank 0)
    data_aux = comm.gather(data, root=0)

    # Root process combines the data and writes to the specified path
    if comm.Get_rank() == 0:
        if not path.exists(path.dirname(filepath)):
            makedirs(path.dirname(filepath), exist_ok=True)
        data_aux = concat(data_aux)
        data_aux.to_file(filepath)

    # Synchronize all processes before returning
    comm.Barrier()

    return True
