#!/usr/bin/env python
from os import path
from hermes_d.config import log_message, record_time, get_time_stamp
from datetime import timedelta, datetime
from datetime import timezone as dtz
from nes import Nes, create_nes, open_netcdf
from mpi4py.MPI import Comm
from configargparse import Namespace
from pandas import DataFrame, to_datetime
from typing import Union, List
from numpy import ndarray


BALANCED = True


def select_grid(comm: Comm, options: Namespace, times: list = None) -> Nes:
    """
    Select or create the destination grid based on configuration options.

    If the grid auxiliary file exists, it will be read; otherwise, it will be created.

    Parameters
    ----------
    comm : Comm
        MPI communicator.
    options : Namespace
        Configuration options.
    times : list of datetime, optional
        List of date times.

    Returns
    -------
    Nes
        Grid NES object with the shapefile & cell_area calculated.

    Raises
    ------
    NotImplementedError
        If the specified grid type is not implemented.

    Notes
    -----
    This function creates a grid based on the specified configuration options, including the domain type,
    resolution, and projection.
    """
    st_time = get_time_stamp()
    if options.domain_name is not None:
        grid_path = path.join(options.auxiliary_files_path, f"grid_{options.domain_name}.nc")
        grid_shp_path = path.join(options.auxiliary_files_path, f"grid_{options.domain_name}.geojson")
    else:
        grid_path = path.join(options.auxiliary_files_path, "grid.nc")
        grid_shp_path = path.join(options.auxiliary_files_path, "grid.geojson")
    log_message('Obtaining grid', level=1)

    if path.exists(grid_path):
        log_message(f'Reading existing grid: {grid_path}', level=2)
        grid = open_netcdf(grid_path, comm=comm, balanced=BALANCED)
        grid.load()
        write_grid = False
    else:
        log_message('Creating grid', level=2)
        write_grid = True
        grid = create_grid(comm, options, times)

    # Add vertical description
    grid.set_levels({"data": options.vertical_description, "units": "m", "positive": "up"})

    # Cell area
    if 'cell_area' not in grid.cell_measures.keys():
        write_grid = True
        grid.calculate_grid_area()
        log_message('Grid: cell area created', level=3)

    # Shapefile
    grid.create_shapefile()
    log_message('Grid: shapefile created', level=3)

    grid.global_attrs['aux_shapefile'] = grid_shp_path
    if not path.exists(grid_shp_path):
        grid.write_shapefile(grid_shp_path)
        log_message(f"Grid: shapefile saved: {grid_shp_path}", level=3)

    # Write
    if write_grid:
        grid.to_netcdf(grid_path, serial=False)
        log_message(f'Grid: file created: {grid_path}', level=2)

    log_message('Destination grid done!', level=2)

    record_time('Init', 'Grid', get_time_stamp() - st_time)
    return grid


def create_grid(comm: Comm, options: Namespace, times: list = None) -> Nes:
    """
    Create a grid object based on the specified options.

    Parameters
    ----------
    comm : Comm
        MPI communicator.
    options : Namespace
        Configuration options.
    times : list of datetime, optional
        List of date times.

    Returns
    -------
    Nes
        Grid NES object.

    Raises
    ------
    NotImplementedError
        If the specified grid type is not implemented.
    """
    st_time = get_time_stamp()
    if options.domain_type == 'global':
        log_message("Global grid with:" +
                    f"\n\t\t\tinc_lat: {options.inc_lat}\n\t\t\tinc_lon: {options.inc_lon}", level=3)
        grid = create_nes(comm=comm, info=False, times=times, projection='global',
                          inc_lat=options.inc_lat, inc_lon=options.inc_lon, balanced=BALANCED)
    elif options.domain_type == 'regular':
        log_message("Creating Regular Lat-Lon grid with:" +
                    f"\n\t\t\tlat_orig: {options.lat_orig}\n\t\t\tlon_orig: {options.lon_orig}" +
                    f"\n\t\t\tinc_lat: {options.inc_lat}\n\t\t\tinc_lon: {options.inc_lon}" +
                    f"\n\t\t\tn_lat: {options.n_lat}\n\t\t\tn_lon: {options.n_lon}", level=3)
        grid = create_nes(comm=comm, info=False, times=times, projection='regular',
                          lat_orig=options.lat_orig, lon_orig=options.lon_orig, inc_lat=options.inc_lat,
                          inc_lon=options.inc_lon, n_lat=options.n_lat, n_lon=options.n_lon, balanced=BALANCED)
        log_message('Regular grid created', level=2)
    elif options.domain_type == 'rotated':
        log_message("Creating Rotated grid selected with:" +
                    f"\n\t\t\tcentre_lat: {options.centre_lat}\n\t\t\tcentre_lon: {options.centre_lon}" +
                    f"\n\t\t\twest_boundary: {options.west_boundary}" +
                    f"\n\t\t\tsouth_boundary: {options.south_boundary}" +
                    f"\n\t\t\tinc_rlat: {options.inc_rlat}\n\t\t\tinc_rlon: {options.inc_rlon}", level=3)
        grid = create_nes(comm=comm, info=False, times=times, projection='rotated',
                          centre_lat=options.centre_lat, centre_lon=options.centre_lon,
                          west_boundary=options.west_boundary, south_boundary=options.south_boundary,
                          inc_rlat=options.inc_rlat, inc_rlon=options.inc_rlon, balanced=BALANCED)
    elif options.domain_type == 'rotated_nested':
        log_message("Creating Rotated-Nested grid with:" +
                    f"\n\t\t\tparent_grid_path: {options.parent_grid_path}" +
                    f"\n\t\t\tparent_ratio: {options.parent_ratio}" +
                    f"\n\t\t\ti_parent_start: {options.i_parent_start}" +
                    f"\n\t\t\tj_parent_start: {options.j_parent_start}" +
                    f"\n\t\t\tn_rlat: {options.n_rlat}\n\t\t\tn_rlon: {options.n_lon}", level=3)
        grid = create_nes(comm=comm, info=False, times=times, projection='rotated-nested',
                          parent_grid_path=options.parent_grid_path, parent_ratio=options.parent_ratio,
                          i_parent_start=options.i_parent_start, j_parent_start=options.j_parent_start,
                          n_rlat=options.n_rlat, n_rlon=options.n_rlon, balanced=BALANCED)
    elif options.domain_type == 'lcc':
        log_message("Creating Lambert Conformal Conic grid with:" +
                    f"\n\t\t\tlat_1: {options.lat_1}\n\t\t\tlat_2: {options.lat_2}" +
                    f"\n\t\t\tlon_0: {options.lon_0}\n\t\t\tlat_0: {options.lat_0}" +
                    f"\n\t\t\tnx: {options.nx}\n\t\t\tny: {options.ny}" +
                    f"\n\t\t\tinc_x: {options.inc_x}\n\t\t\tinc_y: {options.inc_y}" +
                    f"\n\t\t\tx_0: {options.x_0}\n\t\t\ty_0: {options.y_0}", level=3)
        grid = create_nes(comm=comm, info=False, times=times, projection='lcc',
                          lat_1=options.lat_1, lat_2=options.lat_2, lon_0=options.lon_0, lat_0=options.lat_0,
                          nx=options.nx, ny=options.ny, inc_x=options.inc_x, inc_y=options.inc_y,
                          x_0=options.x_0, y_0=options.y_0, balanced=BALANCED)
    elif options.domain_type == 'mercator':
        log_message("Creating Mercator grid with:" +
                    f"\n\t\t\tlat_ts: {options.lat_ts}\n\t\t\tlon_0: {options.lon_0}" +
                    f"\n\t\t\tnx: {options.nx}\n\t\t\tny: {options.ny}" +
                    f"\n\t\t\tinc_x: {options.inc_x}\n\t\t\tinc_y: {options.inc_y}" +
                    f"\n\t\t\tx_0: {options.x_0}\n\t\t\ty_0: {options.y_0}", level=3)
        grid = create_nes(comm=comm, info=False, times=times, projection='mercator',
                          lat_ts=options.lat_ts, lon_0=options.lon_0, nx=options.nx, ny=options.ny,
                          inc_x=options.inc_x, inc_y=options.inc_y, x_0=options.x_0, y_0=options.y_0,
                          balanced=BALANCED)
    else:
        raise NotImplementedError(f"The grid type {options.domain_type} is not implemented. "
                                  "Use 'global', 'regular, 'rotated', 'rotated_nested', 'lcc' or 'mercator.")

    # # add vertical description
    # grid.set_levels({"data": options.vertical_description, "units": "m", "positive": "up"})

    record_time('Init', 'CreateGrid', get_time_stamp() - st_time)
    return grid


def add_local_dates(grid: Nes, date_time: List[datetime], pandas: bool = False) -> Union[DataFrame, ndarray]:
    """
    Add local dates to the grid shapefile based on the specified datetime and timezone information.

    Parameters
    ----------
    grid : Nes
        Grid NES object.
    date_time : list of datetime
        List of date times.
    pandas : bool, optional
        If True, returns a pandas DataFrame; otherwise, returns a NumPy array.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Local date and time information.

    Notes
    -----
    This function parses timezones and adds local dates to the grid shapefile.
    """

    def parse_tz(timezone: str) -> str:
        """
        Parse the timezone (string format).

        It is needed because some libraries have more timezones than others, and it
        tries to simplify setting the strange ones into the nearest common one.

        Parameters
        ----------
        timezone : str
            Not parsed timezone.

        Returns
        -------
        str
            Parsed timezone
        """
        tz_dict = {
            'America/Punta_Arenas': 'America/Santiago',
            'Europe/Astrakhan': 'Europe/Moscow',
            'Asia/Atyrau': 'Asia/Aqtau',
            'Asia/Barnaul': 'Asia/Almaty',
            'Europe/Saratov': 'Europe/Moscow',
            'Europe/Ulyanovsk': 'Europe/Moscow',
            'Europe/Kirov': 'Europe/Moscow',
            'Asia/Tomsk': 'Asia/Novokuznetsk',
            'America/Fort_Nelson': 'America/Vancouver',
            'Asia/Famagusta': 'Asia/Nicosia',
            'America/Nuuk': dtz(timedelta(hours=-3)),
        }

        if timezone in tz_dict:
            timezone = tz_dict[timezone]

        return timezone

    grid.shapefile['local'] = to_datetime(date_time, utc=True)

    def convert_and_localize(group):
        tz = group.name  # This is the 'tzid' value for the current group
        localized = group['local'].dt.tz_convert(tz).dt.tz_localize(None)
        return localized

    def get_offset(group):
        tz = group.name  # This is the 'tzid' value for the current group
        offset = group['local'].dt.tz_localize('UTC').dt.tz_convert(tz).apply(
            lambda x: x.utcoffset().total_seconds() // 3600)
        return offset

    grid.shapefile['local'] = grid.shapefile.groupby('tzid', group_keys=False).apply(convert_and_localize)
    grid.shapefile['offset'] = grid.shapefile.groupby('tzid', group_keys=False).apply(get_offset)

    if pandas:
        time_stamp = grid.shapefile['local'].copy()
    else:
        time_stamp = grid.shapefile['local'].to_numpy().reshape((grid.lat['data'].shape[0], grid.lon['data'].shape[-1]))

    del grid.shapefile['local']
    return time_stamp

# def add_timezone_offset(grid: Nes, date_time: List[datetime], pandas: bool = False) ->Union[pd.DataFrame, np.ndarray]:
#
#     grid.shapefile['local'] = pd.to_datetime(date_time, utc=True)
#
#     grid.shapefile['local'] = grid.shapefile.groupby('tzid')['local'].apply(
#         lambda x: x.dt.tz_convert(x.name).dt.utcoffset()/3600)
