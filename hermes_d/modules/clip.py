from os import path
from re import split
from geopandas import GeoDataFrame, read_file, overlay
from nes import Nes
from shapely.geometry import Point, Polygon
from hermes_d.config import log_message, get_time_stamp, record_time
from typing import Optional, Union


def select_clip(grid: Nes, clip_param: Optional[Union[None, str]] = None) -> GeoDataFrame:
    """
    Create a clipped GeoDataFrame based on the specified parameters.

    Parameters
    ----------
    grid : Nes
        Grid NES object.
    clip_param : None or str, optional
        Indicates the type of clip you want.
        - None: Default clip using the boundaries of the domain.
        - String path: Path to the shapefile that will be used as a clip.
        - String list of points: List of points defining the clipped region.

    Returns
    -------
    GeoDataFrame
        Clipped GeoDataFrame.

    Notes
    -----
    This function creates a GeoDataFrame for clipping based on the specified parameters.

    """
    st_time = get_time_stamp()

    log_message('Obtaining clip', level=1)
    clip = GeoDataFrame(geometry=[grid.shapefile.unary_union.convex_hull], crs=grid.shapefile.crs)

    if clip_param is not None:
        if clip_param[0] == path.sep:
            # Shapefile from path
            log_message('Reading clip from file', level=2)
            log_message('{0}'.format(clip_param), level=3)
            shp = read_file(clip_param)
        else:
            # Shapefile from points
            log_message('Creating clip polygon from points', level=2)
            log_message('{0}'.format(clip_param), level=3)
            shp = get_shp_from_points(clip_param)
        shp.to_crs(clip.crs, inplace=True)

        log_message('Intersection with domain boundaries', level=3)
        geom = overlay(shp, clip, how='intersection', keep_geom_type=False)
        if not geom.empty:
            clip = GeoDataFrame(
                geometry=[geom.unary_union],
                crs=clip.crs)
        else:
            clip = GeoDataFrame(
                geometry=[],
                crs=clip.crs)
    record_time('Init', 'Clip', get_time_stamp() - st_time)
    return clip


def get_shp_from_points(points_str: str) -> GeoDataFrame:
    """
    Create a GeoDataFrame with a single polygon defined by lat-lon points.

    Parameters
    ----------
    points_str : str
        List of points (lat, lon)

    Returns
    -------
    GeoDataFrame
        Clip shapefile.
    """

    str_clip = split(' , | ,|, |,', points_str)
    lon_list = [float(components.split(' ')[0]) for components in str_clip]
    lat_list = [float(components.split(' ')[1]) for components in str_clip]

    if not ((lon_list[0] == lon_list[-1]) and (lat_list[0] == lat_list[-1])):
        lon_list.append(lon_list[0])
        lat_list.append(lat_list[0])

    geom = Polygon([[p.x, p.y] for p in [Point(xy) for xy in zip(lon_list, lat_list)]])
    shp = GeoDataFrame(
        geometry=[geom],
        crs={'init': 'epsg:4326'})

    return shp
