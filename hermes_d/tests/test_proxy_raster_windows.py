"""Regression tests for shared source windows and independent destination ownership."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from hermes_d.modules.proxies import ProxyRaster, ProxyRasterNum, ProxyRasterNumCat, ProxyRasterCat


class TestRasterWindows(unittest.TestCase):
    """Check pixel ownership, empty ranks and conservation through grid clipping."""

    def test_rounding_and_disjoint_partition(self):
        """Cover whole pixels exactly once, including negative offsets and empty ranks."""
        self.assertEqual(ProxyRaster.fix_window(Window(-.3, -.2, 4.3, 2.2)), Window(-1, -1, 6, 4))
        self.assertEqual(ProxyRaster.fix_window(Window(0, 0, 4, 2)), Window(0, 0, 5, 3))
        for count in (1, 8, 32):
            coverage = np.zeros(7, dtype=int)
            for rank in range(count):
                proxy = object.__new__(ProxyRaster)
                proxy.size, proxy.rank = count, rank
                window = proxy.get_partial_window(Window(3, 5, 7, 2))
                start = int(window.col_off) - 3
                coverage[start:start + int(window.width)] += 1
                self.assertEqual(window.row_off, 5)
                self.assertEqual(window.height, 2)
            np.testing.assert_array_equal(coverage, 1)

    def test_readers_conserve_weights_with_remote_destination_cells(self):
        """Conserve per-FID totals with 1, 8 and 32 strips across all three readers."""
        transform = from_origin(4000000, 3002000, 1000, 1000)
        extent = box(4000000, 3000000, 4011000, 3002000)
        grid = gpd.GeoDataFrame({'FID': [10, 20]}, geometry=[
            box(4000250, 3000000, 4004750, 3002000),
            box(4004750, 3000000, 4010750, 3002000)], crs=3035)
        nuts = gpd.GeoDataFrame({'NUTS_ID': ['ES51', 'OTHER']}, geometry=[
            box(4000000, 3000000, 4006500, 3002000),
            box(4006500, 3000000, 4011000, 3002000)], crs=3035).set_index('NUTS_ID')
        population = np.arange(1, 23, dtype=np.float64).reshape(2, 11)
        categories = np.full((2, 11), 11, dtype=np.uint16)
        categories[:, 3] = 12
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('population.tif', 'categories.tif')]
            for path, values in zip(paths, (population, categories)):
                with rasterio.open(path, 'w', driver='GTiff', width=11, height=2, count=1,
                                   crs='EPSG:3035', transform=transform, dtype=values.dtype) as dst:
                    dst.write(values, 1)
            grid_path = Path(directory) / 'grid.geojson'
            grid.to_file(grid_path, driver='GeoJSON')
            for cls in (ProxyRasterNum, ProxyRasterNumCat, ProxyRasterCat):
                baseline = None
                for count in (1, 8, 32):
                    parts, proxies = [], []
                    for rank in range(count):
                        proxy = object.__new__(cls)
                        proxy.size, proxy.rank, proxy.master = count, rank, rank == 0
                        proxy.nuts2_shp_name, proxy.nuts2_shp = 'NUTS_ID', nuts
                        # Only rank 0 owns destination cells; every other strip
                        # must nevertheless intersect and retain these remote FIDs.
                        local_grid = grid if rank == 0 else grid.iloc[:0]
                        proxy.grid = SimpleNamespace(
                            shapefile=local_grid.set_index('FID'), global_attrs={'aux_shapefile': str(grid_path)},
                            get_fids=lambda rank=rank: np.array([10, 20] if rank == 0 else [], dtype=int))
                        proxy.comm = SimpleNamespace(Barrier=lambda: None, bcast=lambda value: value)
                        proxy.get_bbox = lambda full_domain: gpd.GeoDataFrame(geometry=[extent], crs=3035)
                        proxy.category_path, proxy.category_list = str(paths[1]), np.array([11])
                        # Simulate the root broadcast while exercising actual window computation.
                        root_window = Window(0, 0, 11, 2)
                        proxy.comm.bcast = lambda value: root_window
                        with patch('hermes_d.modules.proxies.proxy_raster_categorical.redistribute',
                                   side_effect=lambda data, comm: data):
                            source = proxy.read_src_proxy(str(paths[1] if cls == ProxyRasterCat else paths[0]))
                        fragments = proxy.add_info_to_src_shp(source)
                        self.assertEqual(fragments.crs.to_epsg(), 3035)
                        if not fragments.empty:
                            parts.append(fragments.drop(columns='geometry'))
                        proxies.append((proxy, fragments))
                    all_weights = pd.concat(parts).groupby(['NUTS_ID', 'FID']).weight.sum().sort_index()
                    if baseline is None:
                        baseline = all_weights
                    pd.testing.assert_series_equal(all_weights, baseline, rtol=1e-12, atol=1e-9)
                    gathered = all_weights.rename('weight').reset_index()
                    for proxy, fragments in proxies:
                        proxy.broadcast = lambda data: gathered
                        owned = proxy.get_proxy_by_nut_and_fid(fragments)
                        if proxy.rank == 0:
                            pd.testing.assert_series_equal(owned.weight.sort_index(), baseline,
                                                           rtol=1e-12, atol=1e-9)
                        else:
                            self.assertTrue(owned.empty)

    def test_category_nodata_and_empty_windows(self):
        """Exclude category nodata even if selected, and allow zero-width ranks."""
        with tempfile.TemporaryDirectory() as directory:
            population = Path(directory) / 'population.tif'
            categories = Path(directory) / 'categories.tif'
            for path, values, nodata in ((population, np.array([[7, 9]], dtype=np.uint16), None),
                                         (categories, np.array([[11, 12]], dtype=np.uint16), 11)):
                with rasterio.open(path, 'w', driver='GTiff', width=2, height=1, count=1,
                                   crs='EPSG:3035', transform=from_origin(0, 1, 1, 1),
                                   dtype=values.dtype, nodata=nodata) as dst:
                    dst.write(values, 1)
            for cls in (ProxyRasterNum, ProxyRasterNumCat, ProxyRasterCat):
                proxy = object.__new__(cls)
                proxy.category_path, proxy.category_list = str(categories), np.array([11, 12])
                proxy.comm = SimpleNamespace(Barrier=lambda: None)
                for width in (0, 2):
                    proxy.get_rank_window = lambda path: Window(0, 0, width, 1)
                    with patch('hermes_d.modules.proxies.proxy_raster_categorical.redistribute',
                               side_effect=lambda data, comm: data):
                        source = proxy.read_src_proxy(str(categories if cls == ProxyRasterCat else population))
                    self.assertEqual(source.crs.to_epsg(), 3035)
                    if width == 0:
                        self.assertTrue(source.empty)
                    elif cls == ProxyRasterNum:
                        self.assertEqual(source.weight.sum(), 16)
                    elif cls == ProxyRasterNumCat:
                        self.assertEqual(source.weight.tolist(), [9])
                    else:
                        self.assertEqual(source.geometry.area.sum(), 1)

    def test_total_window_clamps_to_raster_and_checks_category_alignment(self):
        """Clip windows at raster boundaries and reject shifted category pixels."""
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('source.tif', 'shifted.tif')]
            for path, x in zip(paths, (0, .5)):
                with rasterio.open(path, 'w', driver='GTiff', width=4, height=2, count=1,
                                   crs='EPSG:3035', transform=from_origin(x, 2, 1, 1), dtype='uint16') as dst:
                    dst.write(np.ones((2, 4), dtype=np.uint16), 1)
            proxy = object.__new__(ProxyRasterNumCat)
            proxy.master, proxy.rank, proxy.size = True, 0, 1
            proxy.comm = SimpleNamespace(bcast=lambda value: value)
            for bounds, expected in ((box(-10, -10, 10, 10), Window(0, 0, 4, 2)),
                                     (box(10, 10, 11, 11), Window(4, 0, 0, 0))):
                proxy.get_bbox = lambda full_domain: gpd.GeoDataFrame(geometry=[bounds], crs=3035)
                self.assertEqual(proxy.get_total_window(str(paths[0])), expected)
            proxy.category_path, proxy.category_list = str(paths[1]), np.array([1])
            with rasterio.open(paths[0]) as src:
                with self.assertRaisesRegex(RuntimeError, 'identical CRS, transform and dimensions'):
                    proxy.read_window_data(src, Window(0, 0, 4, 2))


if __name__ == '__main__':
    unittest.main()
