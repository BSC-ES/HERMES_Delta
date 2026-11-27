"""Geodesic geometry measurements for spatial proxy weights."""
from geopandas import GeoDataFrame
from pandas import Series
from numpy import divide, zeros
from shapely.geometry.polygon import orient


def get_line_lengths(src_shp: GeoDataFrame) -> Series:
    """Measure line lengths in metres without selecting a regional projection.

    Parameters
    ----------
    src_shp : GeoDataFrame
        Line geometries with a defined geographic or projected CRS.

    Returns
    -------
    pandas.Series
        Geodesic lengths in metres, indexed like ``src_shp``. Missing and
        empty geometries have zero length.

    Raises
    ------
    ValueError
        If ``src_shp`` has no CRS.

    Notes
    -----
    A temporary geometry series is transformed to the geographic CRS associated
    with the input CRS. Measurements use its ellipsoid. The input geometries
    and their CRS are not modified.
    """
    if src_shp.crs is None:
        raise ValueError('A CRS is required to calculate geodesic line lengths.')
    geographic_crs = src_shp.crs.geodetic_crs
    geod = geographic_crs.get_geod()
    geometries = src_shp.geometry.to_crs(geographic_crs)
    return geometries.apply(
        lambda geometry: 0.0 if geometry is None or geometry.is_empty else geod.geometry_length(geometry))


def get_polygon_areas(src_shp: GeoDataFrame) -> Series:
    """Measure polygon areas in square metres on the input CRS ellipsoid.

    Parameters
    ----------
    src_shp : GeoDataFrame
        Polygon geometries with a defined geographic or projected CRS.

    Returns
    -------
    pandas.Series
        Nonnegative geodesic areas in square metres, indexed like ``src_shp``.
        Missing, empty and non-polygonal geometries have zero area.

    Raises
    ------
    ValueError
        If ``src_shp`` has no CRS.

    Notes
    -----
    A temporary geometry series is transformed to the geographic CRS associated
    with the input CRS. Ring orientation is normalized so holes subtract from
    exterior areas. Multipart polygons and polygonal members of geometry
    collections are measured independently and their areas summed.

    Each polygon must cover less than half the ellipsoid surface. Geodesic
    edges follow arcs between consecutive vertices, so fragmenting planar
    geometries can introduce small differences in total measured area.
    The input geometries and their CRS are not modified.
    """
    if src_shp.crs is None:
        raise ValueError('A CRS is required to calculate geodesic polygon areas.')
    geographic_crs = src_shp.crs.geodetic_crs
    geod = geographic_crs.get_geod()
    geometries = src_shp.geometry.to_crs(geographic_crs)

    def polygon_area(geometry):
        """Measure the polygonal area of one geographic geometry.

        Parameters
        ----------
        geometry : shapely.geometry.base.BaseGeometry or None
            Geometry expressed in the geographic CRS prepared by the caller.

        Returns
        -------
        float
            Unsigned polygonal area in square metres. Missing, empty and
            non-polygonal geometries contribute zero.
        """
        if geometry is None or geometry.is_empty:
            return 0.0
        if geometry.geom_type == 'Polygon':
            return abs(geod.geometry_area_perimeter(orient(geometry, sign=1.0))[0])
        if geometry.geom_type in ('MultiPolygon', 'GeometryCollection'):
            return sum(polygon_area(part) for part in geometry.geoms)
        return 0.0

    return geometries.apply(polygon_area)


def get_polygon_area_fractions(source: GeoDataFrame, intersections: GeoDataFrame):
    """Calculate intersection-to-source fractions using geodesic polygon areas.

    Parameters
    ----------
    source : GeoDataFrame
        Source polygons with a defined CRS and a unique index.
    intersections : GeoDataFrame
        Intersection geometries with a defined CRS. The ``gdf1_idx`` column
        contains the corresponding source index labels.

    Returns
    -------
    numpy.ndarray
        Dimensionless fractions in intersection row order. Zero-area sources
        produce zero fractions.

    Raises
    ------
    ValueError
        If an input has no CRS.
    KeyError
        If ``gdf1_idx`` is missing or contains unknown source labels.

    Notes
    -----
    Each source is measured once. Intersections are transformed to the source
    CRS so both measures use the same ellipsoid. Fractions are not normalized:
    partial destination coverage may give a sum below one. Neither input is
    modified. Polygon size restrictions and fragmentation tolerances follow
    ``get_polygon_areas``.
    """
    areas = get_polygon_areas(source)
    denominators = areas.loc[intersections['gdf1_idx']].to_numpy(dtype=float)
    numerators = get_polygon_areas(intersections.to_crs(source.crs)).to_numpy(dtype=float)
    return divide(numerators, denominators, out=zeros(len(numerators)), where=denominators > 0)


def get_line_length_fractions(source: GeoDataFrame, intersections: GeoDataFrame):
    """Calculate intersection-to-source fractions using geodesic line lengths.

    Parameters
    ----------
    source : GeoDataFrame
        Source lines with a defined CRS and a unique index.
    intersections : GeoDataFrame
        Intersection geometries with a defined CRS. The ``gdf1_idx`` column
        contains the corresponding source index labels.

    Returns
    -------
    numpy.ndarray
        Dimensionless fractions in intersection row order. Zero-length sources
        produce zero fractions.

    Raises
    ------
    ValueError
        If an input has no CRS.
    KeyError
        If ``gdf1_idx`` is missing or contains unknown source labels.

    Notes
    -----
    Each source is measured once. Intersections are transformed to the source
    CRS so both measures use the same ellipsoid. Fractions are not normalized,
    allowing partial coverage. Neither input is modified.
    """
    lengths = get_line_lengths(source)
    denominators = lengths.loc[intersections['gdf1_idx']].to_numpy(dtype=float)
    numerators = get_line_lengths(intersections.to_crs(source.crs)).to_numpy(dtype=float)
    return divide(numerators, denominators, out=zeros(len(numerators)), where=denominators > 0)


def get_raster_polygon_areas(src_shp: GeoDataFrame) -> Series:
    """Measure areas in a raster's native CRS without selecting a new projection.

    Parameters
    ----------
    src_shp : GeoDataFrame
        Polygon geometries expressed in the native raster CRS.

    Returns
    -------
    pandas.Series
        Areas in square metres, indexed like the input. Projected CRS areas
        are planar, converted using the two axis unit conversion factors.
        Geographic CRS areas are measured geodesically.

    Raises
    ------
    ValueError
        If the CRS is missing or is neither projected nor geographic.

    Notes
    -----
    Planar areas are additive for a fixed projected geometry. Their physical
    distortion depends on the native projection; a metric CRS is not necessarily
    equal-area. Geographic rasters retain geodesic measurement to avoid square
    degrees and therefore retain its fragmentation limitations. Inputs are not
    modified and no automatic regional projection is chosen.
    """
    if src_shp.crs is None:
        raise ValueError('A CRS is required to calculate raster polygon areas.')
    if src_shp.crs.is_geographic:
        return get_polygon_areas(src_shp)
    if not src_shp.crs.is_projected:
        raise ValueError('Raster areas require a projected or geographic CRS.')
    axes = src_shp.crs.axis_info
    factor = axes[0].unit_conversion_factor * axes[1].unit_conversion_factor
    return src_shp.geometry.area.fillna(0.) * factor


def get_raster_area_fractions(source: GeoDataFrame, intersections: GeoDataFrame):
    """Calculate area fractions consistently in the native raster CRS.

    Parameters
    ----------
    source : GeoDataFrame
        Source polygons in the native raster CRS with a unique index.
    intersections : GeoDataFrame
        Fragment geometries with a defined CRS and a ``gdf1_idx`` column
        identifying the corresponding source index labels.

    Returns
    -------
    numpy.ndarray
        Dimensionless fractions in fragment row order. Zero-area sources
        produce zero fractions; partial coverage is not normalized to one.

    Raises
    ------
    ValueError
        If the CRS is missing or unsupported.
    KeyError
        If source labels or ``gdf1_idx`` are missing.

    Notes
    -----
    Source and fragment areas use ``get_raster_polygon_areas`` in the same CRS.
    Each source area is calculated once. Neither input is modified.
    """
    areas = get_raster_polygon_areas(source)
    denominators = areas.loc[intersections['gdf1_idx']].to_numpy(dtype=float)
    numerators = get_raster_polygon_areas(intersections.to_crs(source.crs)).to_numpy(dtype=float)
    return divide(numerators, denominators, out=zeros(len(numerators)), where=denominators > 0)
