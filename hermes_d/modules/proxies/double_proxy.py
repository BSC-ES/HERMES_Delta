from hermes_d.config import precision
from geopandas import read_file, GeoDataFrame
from numpy import array
from nes import Nes
from typing import Optional
from hermes_d.modules.proxies import Proxy


class DoubleProxy(object):
    def __init__(self, first_proxy: Proxy, second_proxy: dict):
        """
        Initialize a DoubleProxy instance.

        Parameters
        ----------
        first_proxy : Proxy
            The first proxy instance.
        second_proxy : dict
            Dictionary containing information about the second proxy.
        """
        self.first_proxy = first_proxy
        self.second_proxy = second_proxy

        self.update_first_proxy()

    def read_src_proxy(self, path: str, clip: Optional[GeoDataFrame] = None) -> GeoDataFrame:
        """
        Read the source proxy and return it as a shapefile.

        Parameters
        ----------
        path : str
            Path to the source proxy file.
        clip : GeoDataFrame, optional
            Region of the raster to read.

        Returns
        -------
        GeoDataFrame
            Source proxy shapefile.
        """
        if clip is None:
            clip = self.first_proxy.clip

        if len(clip) > 0:
            # Read and filter shapefile based on provided clip region
            proxy_shp = read_file(path, bbox=clip).to_crs(clip.crs)
            proxy_shp['weight'] = array(proxy_shp['weight'], dtype=precision)
            proxy_shp = proxy_shp.loc[proxy_shp[self.second_proxy["column"]] == self.second_proxy["name"],
                                      ['weight', 'geometry']]
        else:
            # Empty proxy if clip is not provided
            proxy_shp = GeoDataFrame(columns=['weight'], geometry=[], crs=self.first_proxy.clip.crs)

        return proxy_shp

    def update_first_proxy(self) -> None:
        """
        Update the first proxy based on the second proxy information.

        Returns
        -------
        None
        """
        self.first_proxy.src_shp = self.first_proxy.src_shp.reset_index(drop=True)

        if self.first_proxy.src_shp.empty:
            return None
        second_shape = self.read_src_proxy(path=self.second_proxy['proxy_data'], clip=self.first_proxy.src_shp)

        if second_shape.empty:
            self.first_proxy.src_shp = GeoDataFrame()
            return None

        second_shape = second_shape.to_crs(self.first_proxy.src_shp.crs)
        second_shape = second_shape.reset_index(drop=True)

        # Reset index
        # gdf2_idx, gdf1_idx = self.first_proxy.src_shp.sindex.query_bulk(second_shape.geometry, predicate='intersects')
        gdf2_idx, gdf1_idx = self.first_proxy.src_shp.sindex.query(second_shape.geometry, predicate='intersects')

        if len(gdf2_idx) > 0:
            # Remove duplicates from the original shp index and their corresponding elements from the second shp index
            gdf1_idx, gdf2_idx = remove_duplicates_and_corresponding(gdf1_idx, gdf2_idx)

            # Keeping only the intersected source proxy
            self.first_proxy.src_shp = self.first_proxy.src_shp.loc[gdf1_idx]

            # Weighting the first proxy with the first intersection of the second one
            try:
                self.first_proxy.src_shp.loc[gdf1_idx, 'weight'] *= second_shape.loc[gdf2_idx, 'weight'].to_numpy()
            except Exception as e:
                msg = f"FAIL rank {self.first_proxy.rank}\n"
                msg += f"{gdf2_idx}\n"
                msg += f"{second_shape}"
                print(msg)
                raise e
            # Dropping proxy with no weight
            self.first_proxy.src_shp = self.first_proxy.src_shp.loc[self.first_proxy.src_shp['weight'] > 0]
        else:
            # If no intersection remove all the content
            self.first_proxy.src_shp = self.first_proxy.src_shp.drop(self.first_proxy.src_shp.index)
        return None

    def add_proxy_to_grid(self, proxy_name: str, totals_path: str, proxy_out: Nes) -> Nes:
        """
        Add the proxy to the grid. Uses the first proxy method

        Parameters
        ----------
        proxy_name : str
            Name of the proxy.
        proxy_out : Nes
            A nes object to add the proxy

        Notes
        -----
        - This method adds a new variable to the grid with the specified proxy_name.
        - The variable is initialized with zeros and has dimensions based on the grid and NUTS2 list.
        - For each NUTS2 region, it extracts data from the source proxy shapefile and assigns it to the grid variable.
        - The data is weighted by the 'weight' column in the source proxy shapefile.
        - The result is stored in the grid variable with the specified proxy_name.
        """
        return self.first_proxy.add_proxy_to_grid(proxy_name, totals_path, proxy_out)


def remove_duplicates_and_corresponding(list1: list, list2: list) -> tuple:
    """
    Remove duplicates from the first list and their corresponding elements from the second list.

    Parameters
    ----------
    list1 : list
        The first list with potentially duplicate elements.
    list2 : list
        The second list with corresponding elements.

    Returns
    -------
    Tuple
        A tuple containing two NumPy arrays - the filtered unique elements from list1 and the corresponding elements
        from list2.

    Examples
    --------
    >>> list_1 = [1, 2, 3, 1, 2, 4, 5]
    >>> list_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    >>> filtered_list_1, filtered_list_2 = remove_duplicates_and_corresponding(list_1, list_2)
    >>> print(filtered_list_1)
    array([1, 2, 3, 4, 5])
    >>> print(filtered_list_2)
    array(['a', 'b', 'c', 'f', 'g'])
    """
    unique_elements = set()
    filtered_lists = zip(*((el1, el2) for el1, el2 in zip(list1, list2) if
                           el1 not in unique_elements and (unique_elements.add(el1) or True)))

    # Convert each tuple to a NumPy array
    filtered_list1, filtered_list2 = (array(arr) for arr in filtered_lists)

    return filtered_list1, filtered_list2
