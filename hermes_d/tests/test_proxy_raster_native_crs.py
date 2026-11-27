"""Regression tests for raster-native spatial calculations in v1.0.7b2."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from hermes_d.modules.proxies import ProxyRasterNum, ProxyRasterNumCat, ProxyRasterCat, ProxyShpPoly
from hermes_d.utilities import get_raster_polygon_areas, get_raster_area_fractions, get_polygon_areas


class TestRasterNativeCRS(unittest.TestCase):
    """Check raster CRS preservation, area conservation and grid reprojection."""

    def test_all_readers_and_overlays_use_native_crs(self):
        """Keep raster footprints while intersecting grids in different projections."""
        native = CRS.from_epsg(3035)
        extent = box(4000000, 3000000, 4004000, 3002000)
        native_grid = gpd.GeoDataFrame({'FID': [10, 20]}, geometry=[
            box(4000000, 3000000, 4001500, 3002000), box(4001500, 3000000, 4004000, 3002000)],
            crs=native).set_index('FID')
        rotated = CRS.from_proj4('+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=45 +lon_0=0 +datum=WGS84')
        clip = gpd.GeoDataFrame(geometry=[extent], crs=native).to_crs(4326)
        nuts = gpd.GeoDataFrame({'NUTS_ID': ['R']}, geometry=[extent.buffer(1000)], crs=native).to_crs(4326)
        nuts = nuts.set_index('NUTS_ID')
        with tempfile.TemporaryDirectory() as directory:
            population = Path(directory) / 'population.tif'
            categories = Path(directory) / 'categories.tif'
            for path, values in ((population, np.full((2, 4), 2.5)),
                                 (categories, np.full((2, 4), 11, dtype=np.uint16))):
                with rasterio.open(path, 'w', driver='GTiff', width=4, height=2, count=1,
                                   dtype=values.dtype, crs=native,
                                   transform=from_origin(4000000, 3002000, 1000, 1000)) as dst:
                    dst.write(values, 1)
            for grid_crs in (4326, 3857, rotated):
                for cls in (ProxyRasterNum, ProxyRasterNumCat, ProxyRasterCat):
                    with self.subTest(grid_crs=grid_crs, reader=cls.__name__):
                        grid = native_grid.to_crs(grid_crs)
                        proxy = object.__new__(cls)
                        proxy.grid = SimpleNamespace(shapefile=grid)
                        proxy.clip = clip
                        proxy.category_path = str(categories)
                        proxy.category_list = np.array([11])
                        proxy.comm = SimpleNamespace(Barrier=lambda: None)
                        proxy.get_rank_window = lambda path: Window(0, 0, 4, 2)
                        with patch('hermes_d.modules.proxies.proxy_raster_categorical.redistribute',
                                   side_effect=lambda data, comm: data):
                            source = proxy.read_src_proxy(str(categories if cls == ProxyRasterCat else population))
                        self.assertEqual(source.crs, native)
                        self.assertTrue(source.geometry.union_all().equals(extent))
                        result = proxy.add_nut_codes(source, nuts)
                        result = proxy.add_fid(result, grid)
                        self.assertEqual(result.crs, native)
                        # Compare with the actual transformed cell footprints:
                        # LAEA forward/inverse transformations have finite precision.
                        expected_areas = grid.to_crs(native).geometry.intersection(extent).area
                        if cls == ProxyRasterCat:
                            result['weight'] = get_raster_polygon_areas(result)
                            np.testing.assert_allclose(result.weight.sum(), expected_areas.sum(), rtol=1e-10)
                        else:
                            expected_weights = expected_areas * 2 / 1000000
                            np.testing.assert_allclose(result.weight.sum(), expected_weights.sum(), rtol=1e-10)
                            by_fid = result.groupby('FID').weight.sum()
                            np.testing.assert_allclose(by_fid.loc[[10, 20]], expected_weights.loc[[10, 20]], rtol=1e-10)

    def test_categorical_reads_grid_bbox_in_grid_crs(self):
        """Read auxiliary geographic grid candidates using a transformed raster bbox."""
        source = gpd.GeoDataFrame({'weight': [1.]},
                                  geometry=[box(4000000, 3000000, 4002000, 3001000)], crs=3035)
        grid = gpd.GeoDataFrame({'FID': [4]}, geometry=source.geometry, crs=3035).to_crs(4326)
        nuts = gpd.GeoDataFrame({'NUTS_ID': ['R']}, geometry=source.geometry.buffer(1000),
                                crs=3035).set_index('NUTS_ID')
        proxy = object.__new__(ProxyRasterCat)
        proxy.nuts2_shp = nuts.to_crs(4326)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grid.geojson'
            grid.to_file(path, driver='GeoJSON')
            proxy.grid = SimpleNamespace(shapefile=grid, global_attrs={'aux_shapefile': str(path)})
            with patch('hermes_d.modules.proxies.proxy_raster.read_file', wraps=gpd.read_file):
                result = proxy.add_info_to_src_shp(source)
        self.assertFalse(result.empty)
        self.assertEqual(result.crs.to_epsg(), 3035)
        expected = source.geometry.iloc[0].intersection(grid.to_crs(source.crs).geometry.iloc[0]).area
        np.testing.assert_allclose(result.weight.sum(), expected, rtol=1e-10)

    def test_projected_area_conservation_and_partial_fractions(self):
        """Conserve fixed projected footprints with 1, 8 and 32 subdivisions."""
        outer = box(0, 0, 32000, 8000)
        source = gpd.GeoDataFrame(geometry=[outer], crs=3035)
        for count in (1, 8, 32):
            parts = gpd.GeoDataFrame({'gdf1_idx': [0] * count}, geometry=[
                box(32000 * i / count, 0, 32000 * (i + 1) / count, 8000) for i in range(count)], crs=3035)
            self.assertEqual(get_raster_polygon_areas(parts).sum(), outer.area)
            self.assertAlmostEqual(get_raster_area_fractions(source, parts).sum(), 1.)
        partial = gpd.GeoDataFrame({'gdf1_idx': [0]}, geometry=[box(0, 0, 16000, 8000)], crs=3035)
        np.testing.assert_allclose(get_raster_area_fractions(source, partial), [.5])

    def test_categorical_windows_preserve_native_area(self):
        """Preserve categorical area per FID across 1, 8 and 32 raster strips."""
        native = gpd.GeoDataFrame({'FID': [1, 2]}, geometry=[
            box(4000000, 3000000, 4003150, 3000800),
            box(4003150, 3000000, 4006400, 3000800)], crs=3035).set_index('FID')
        grid = native.to_crs(3857)
        nuts = gpd.GeoDataFrame({'NUTS_ID': ['R']}, geometry=[
            box(3999000, 2999000, 4007400, 3001800)], crs=3035).to_crs(4326).set_index('NUTS_ID')
        values = np.full((8, 64), 11, dtype=np.uint16)
        values[2:6, 20:37] = 0
        baseline = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'categories.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=64, height=8, count=1,
                               dtype=values.dtype, crs='EPSG:3035',
                               transform=from_origin(4000000, 3000800, 100, 100)) as dst:
                dst.write(values, 1)
            for count in (1, 8, 32):
                totals = np.zeros(2)
                for rank in range(count):
                    proxy = object.__new__(ProxyRasterCat)
                    proxy.rank, proxy.size = rank, count
                    proxy.comm = SimpleNamespace(Barrier=lambda: None)
                    proxy.category_list = np.array([11])
                    proxy.get_rank_window = lambda path: proxy.get_partial_window_x(Window(0, 0, 64, 8))
                    with patch('hermes_d.modules.proxies.proxy_raster_categorical.redistribute',
                               side_effect=lambda data, comm: data):
                        source = proxy.read_src_proxy(str(path))
                    source = proxy.add_nut_codes(source, nuts)
                    source = proxy.add_fid(source, grid)
                    source['weight'] = get_raster_polygon_areas(source)
                    totals += source.groupby('FID').weight.sum().reindex([1, 2], fill_value=0).to_numpy()
                if baseline is None:
                    baseline = totals
                np.testing.assert_allclose(totals, baseline, rtol=1e-12, atol=1e-6)

    def test_geographic_fallback_and_projected_units(self):
        """Avoid square degrees and convert projected feet to square metres."""
        geographic = gpd.GeoDataFrame(geometry=[box(2, 41, 2.1, 41.1)], crs=4326)
        np.testing.assert_allclose(get_raster_polygon_areas(geographic), get_polygon_areas(geographic))
        feet = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 20)], crs=2263)
        factor = feet.crs.axis_info[0].unit_conversion_factor
        np.testing.assert_allclose(get_raster_polygon_areas(feet), [200 * factor ** 2])
        with self.assertRaisesRegex(ValueError, 'CRS is required'):
            get_raster_polygon_areas(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]))

    def test_shapefile_polygon_keeps_geodesic_fraction_default(self):
        """Leave shapefile proxy area policy independent of the raster experiment."""
        source = gpd.GeoDataFrame({'weight': [100.]}, geometry=[box(2, 40, 2.01, 42)], crs=4326)
        cells = gpd.GeoDataFrame({'FID': [1, 2]}, geometry=[
            box(2, 40, 2.01, 41), box(2, 41, 2.01, 42)], crs=4326)
        result = ProxyShpPoly.overlay(source, cells, one_intersection=True, weight_vars='weight')
        self.assertGreater(result.loc[result.FID == 1, 'weight'].iloc[0], 50.)


if __name__ == '__main__':
    unittest.main()
