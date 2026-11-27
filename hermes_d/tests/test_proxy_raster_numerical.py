"""Regression tests for deterministic numerical raster pixel weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from hermes_d.modules.proxies import ProxyRasterNum, ProxyRasterNumCat


class TestNumericalPixelWeights(unittest.TestCase):
    """Verify that pixel separation never modifies numerical proxy values."""

    def test_equal_neighbours_and_large_values_keep_original_weights(self):
        """Preserve counts without merging equal pixels or rounding through float32."""
        data = np.array([[166.934031, 166.448316, 1000., 16777217., 100000000.]], dtype=np.float64)
        original = data.copy()
        result = ProxyRasterNum.polygonize_pixels(data, from_origin(0, 1, 1, 1), 'EPSG:4326')
        self.assertEqual(len(result), data.size)
        np.testing.assert_array_equal(sorted(result.weight), sorted(np.floor(data).ravel()))
        self.assertEqual(int(result.weight.sum()), 116778549)
        np.testing.assert_array_equal(data, original)

    def test_invalid_masked_and_empty_pixels_are_excluded(self):
        """Exclude nodata and nonpositive pixels without reading masked payloads."""
        data = np.ma.array([[2.8, 0.9, -2., np.nan, np.inf, 5.]], mask=[[0, 0, 0, 0, 0, 1]])
        result = ProxyRasterNum.polygonize_pixels(data, Affine.identity(), 'EPSG:4326')
        self.assertEqual(result.weight.tolist(), [2])
        for empty in (np.zeros((2, 2)), np.empty((0, 0)), np.ma.masked_all((2, 2))):
            result = ProxyRasterNum.polygonize_pixels(empty, Affine.identity(), 'EPSG:4326')
            self.assertTrue(result.empty)
            self.assertEqual(result.crs.to_epsg(), 4326)
            self.assertIn('weight', result)

    def test_window_partition_and_rotated_transform_preserve_pixels(self):
        """Generate the same footprints and weights from whole or split windows."""
        data = np.array([[173., 173., 1000., 1000.], [4., 4., 4., 4.]])
        transform = Affine(1000, 50, 400000, 30, -1000, 4600000)
        whole = ProxyRasterNum.polygonize_pixels(data, transform, 'EPSG:25831')
        parts = [ProxyRasterNum.polygonize_pixels(data[:, start:start + 2],
                 transform * Affine.translation(start, 0), 'EPSG:25831') for start in (0, 2)]
        whole_by_geometry = {geometry.wkb: int(weight) for geometry, weight in zip(whole.geometry, whole.weight)}
        split_by_geometry = {geometry.wkb: int(weight) for part in parts
                             for geometry, weight in zip(part.geometry, part.weight)}
        self.assertEqual(whole_by_geometry, split_by_geometry)

    def test_both_readers_are_independent_of_random_state(self):
        """Read identical numerical and categorized inputs consistently across seeds."""
        data = np.array([[166.9, 166.4, 1000., 100000000.], [5., 0., -9999., 16777217.]])
        categories = np.array([[11, 11, 12, 11], [11, 11, 11, 12]], dtype=np.uint16)
        transform = from_origin(2, 42, .01, .01)
        clip = gpd.GeoDataFrame(geometry=[box(2, 41.98, 2.04, 42)], crs=4326)
        with tempfile.TemporaryDirectory() as directory:
            population_path = Path(directory) / 'population.tif'
            category_path = Path(directory) / 'categories.tif'
            for path, values, nodata in ((population_path, data, -9999.), (category_path, categories, None)):
                with rasterio.open(path, 'w', driver='GTiff', height=2, width=4, count=1,
                                   dtype=values.dtype, crs='EPSG:4326', transform=transform, nodata=nodata) as dst:
                    dst.write(values, 1)
            for cls, expected in ((ProxyRasterNum, [166, 166, 1000, 100000000, 5, 16777217]),
                                  (ProxyRasterNumCat, [166, 166, 100000000, 5])):
                proxy = object.__new__(cls)
                proxy.get_rank_window = lambda path: Window(0, 0, 4, 2)
                proxy.clip = clip
                proxy.grid = SimpleNamespace(shapefile=clip)
                proxy.category_path = str(category_path)
                proxy.category_list = np.array([11])
                baseline = None
                for seed in (0, 1, 3, 7):
                    with self.subTest(reader=cls.__name__, seed=seed):
                        np.random.seed(seed)
                        before = np.random.get_state()
                        result = proxy.read_src_proxy(str(population_path))
                        after = np.random.get_state()
                        np.testing.assert_array_equal(before[1], after[1])
                        self.assertEqual(before[2:], after[2:])
                        np.testing.assert_array_equal(sorted(result.weight), sorted(expected))
                        if baseline is not None:
                            self.assertTrue(result.equals(baseline))
                        baseline = result


if __name__ == '__main__':
    unittest.main()
