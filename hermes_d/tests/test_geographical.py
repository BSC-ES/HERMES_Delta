"""Regression tests for shared geodesic measurements."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from pyproj import Geod
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, box
from shapely.geometry.polygon import orient

from hermes_d.utilities import get_line_lengths, get_polygon_areas
from hermes_d.modules.proxies import ProxyRasterCat
from hermes_d.modules.proxies import ProxyShpPoly, ProxyRasterNum, ProxyRasterNumCat, ProxyShpLine
from hermes_d.utilities import get_polygon_area_fractions, get_line_length_fractions


class TestGeographical(unittest.TestCase):
    """Verify physical units, geometry handling and categorical proxy weights."""

    def test_line_lengths_are_metres_for_both_orientations(self):
        """Measure equal physical lengths in geographic and projected inputs."""
        geod = Geod(ellps='WGS84')
        start = (2., 41.5)
        source = gpd.GeoDataFrame(geometry=[
            LineString([start, geod.fwd(*start, azimuth, 10000)[:2]]) for azimuth in (0, 90)], crs=4326)
        for crs in (4326, 25831):
            with self.subTest(crs=crs):
                np.testing.assert_allclose(get_line_lengths(source.to_crs(crs)), [10000., 10000.], atol=1e-6)

    def test_polygon_areas_handle_holes_and_multipart_orientation(self):
        """Subtract holes and add multipart areas regardless of ring orientation."""
        outer = box(2, 41, 2.02, 41.02)
        hole = box(2.005, 41.005, 2.015, 41.015)
        polygon = Polygon(outer.exterior.coords, [hole.exterior.coords])
        other = box(2.03, 41, 2.04, 41.01)
        geod = Geod(ellps='WGS84')
        expected = abs(geod.geometry_area_perimeter(outer)[0]) - abs(geod.geometry_area_perimeter(hole)[0])
        other_area = abs(geod.geometry_area_perimeter(other)[0])
        geometries = [polygon, orient(polygon, -1), MultiPolygon([polygon, orient(other, -1)])]
        source = gpd.GeoDataFrame(geometry=geometries, crs=4326)
        for crs in (4326, 25831):
            with self.subTest(crs=crs):
                projected = source.to_crs(crs)
                original = projected.copy()
                np.testing.assert_allclose(get_polygon_areas(projected),
                                           [expected, expected, expected + other_area], atol=0.01)
                self.assertTrue(projected.equals(original))

    def test_empty_and_missing_geometries(self):
        """Return zero for missing geometries and preserve input indices."""
        for measure, geometry in ((get_line_lengths, LineString()), (get_polygon_areas, Polygon())):
            with self.subTest(measure=measure.__name__):
                source = gpd.GeoDataFrame(geometry=[geometry, None], index=[7, 3], crs=4326)
                result = measure(source)
                np.testing.assert_array_equal(result, [0., 0.])
                self.assertTrue(result.index.equals(source.index))
                self.assertTrue(measure(source.iloc[:0]).empty)
                with self.assertRaisesRegex(ValueError, 'CRS is required'):
                    measure(gpd.GeoDataFrame(geometry=[geometry]))

    def test_categorical_proxy_uses_square_metres(self):
        """Assign physical areas to categorical fragments at different latitudes."""
        source = gpd.GeoDataFrame({'weight': [1., 1.]},
                                  geometry=[box(2, 40, 2.01, 40.01), box(2, 42, 2.01, 42.01)], crs=4326)
        nuts = gpd.GeoDataFrame({'NUTS_ID': ['region']}, geometry=[box(1, 39, 3, 43)], crs=4326)
        grid = gpd.GeoDataFrame({'FID': [0]}, geometry=[box(1, 39, 3, 43)], crs=4326)
        proxy = object.__new__(ProxyRasterCat)
        proxy.nuts2_shp_name = 'NUTS_ID'
        proxy.nuts2_shp_path = 'nuts.shp'
        proxy.nuts2_shp = nuts.set_index('NUTS_ID')
        proxy.grid = SimpleNamespace(global_attrs={'aux_shapefile': 'grid.shp'}, shapefile=grid)
        with patch('hermes_d.modules.proxies.proxy_raster.read_file',
                   return_value=grid.copy()):
            result = proxy.add_info_to_src_shp(source.copy())
        np.testing.assert_allclose(result.weight, get_polygon_areas(source))
        self.assertGreater(result.weight.iloc[0], result.weight.iloc[1])

    def test_fractions_preserve_partial_coverage_and_source_labels(self):
        """Use physical measures and source labels without normalizing partial coverage."""
        for measure, fraction, geometries in (
                (get_polygon_areas, get_polygon_area_fractions,
                 [box(2, 40, 2.01, 42), box(2, 40, 2.01, 41)]),
                (get_line_lengths, get_line_length_fractions,
                 [LineString([(2, 40), (2, 42)]), LineString([(2, 40), (2, 41)])])):
            with self.subTest(fraction=fraction.__name__):
                source = gpd.GeoDataFrame(geometry=[geometries[0], geometries[1]], index=[9, 4], crs=4326)
                fragments = gpd.GeoDataFrame({'gdf1_idx': [4, 9]},
                                             geometry=[geometries[1], geometries[1]], crs=4326)
                expected = [1., measure(fragments).iloc[1] / measure(source).loc[9]]
                result = fraction(source, fragments.to_crs(25831))
                np.testing.assert_allclose(result, expected, atol=1e-8)
                self.assertLess(result[1], 1.)
                empty = fragments.iloc[:0]
                self.assertEqual(len(fraction(source, empty)), 0)
                zero = gpd.GeoDataFrame(geometry=[Point(2, 41)], index=[4], crs=4326)
                np.testing.assert_array_equal(fraction(zero, fragments.iloc[:1]), [0.])
                with self.assertRaisesRegex(ValueError, 'CRS is required'):
                    fraction(gpd.GeoDataFrame(geometry=source.geometry.to_list()), fragments)

    def test_polygon_overlays_conserve_weights_across_resolutions(self):
        """Conserve physical weights in every polygon overlay at 10 km and 1 km."""
        source = gpd.GeoDataFrame({'weight': [123.]}, geometry=[
            box(400500, 4600500, 419500, 4619500)], crs=25831).to_crs(4326)
        for proxy_class in (ProxyShpPoly, ProxyRasterNum, ProxyRasterNumCat, ProxyRasterCat):
            for resolution in (10000, 1000):
                with self.subTest(proxy=proxy_class.__name__, resolution=resolution):
                    geometries = [box(400000 + x, 4600000 + y,
                                      400000 + x + resolution, 4600000 + y + resolution)
                                  for y in range(0, 20000, resolution) for x in range(0, 20000, resolution)]
                    cells = gpd.GeoDataFrame({'FID': range(len(geometries))}, geometry=geometries,
                                             crs=25831).to_crs(4326)
                    result = proxy_class.overlay(source, cells, one_intersection=True, weight_vars='weight')
                    np.testing.assert_allclose(result.weight.sum(), 123., rtol=1e-5)

    def test_line_overlay_uses_lengths(self):
        """Split line weights by length instead of dividing zero polygon areas."""
        source = gpd.GeoDataFrame({'weight': [100.]}, geometry=[
            LineString([(2, 41), (2, 41.02)])], crs=4326)
        cells = gpd.GeoDataFrame({'FID': [0, 1]}, geometry=[
            box(1.9, 40.99, 2.1, 41.005), box(1.9, 41.005, 2.1, 41.03)], crs=4326)
        result = ProxyShpLine.overlay(source, cells, one_intersection=True, weight_vars='weight')
        np.testing.assert_allclose(result.weight.sum(), 100., atol=1e-6)
        np.testing.assert_allclose(result.sort_values('FID').weight, [25., 75.], atol=0.001)


if __name__ == '__main__':
    unittest.main()
