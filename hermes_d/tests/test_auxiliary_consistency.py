#!/usr/bin/env python

# Copyright 2026 Earth Sciences Department, BSC-CNS
#
# This file is part of HERMESv3.
#
# HERMESv3 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HERMESv3 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HERMESv3. If not, see <http://www.gnu.org/licenses/>.

import unittest
from types import SimpleNamespace

import numpy as np

from hermes_d.tools.auxiliary_consistency import check_aux_file
from hermes_d.tools.auxiliary_consistency import check_grid
from hermes_d.tools.auxiliary_consistency import check_grid_geostructure
from hermes_d.tools.auxiliary_consistency import get_expected_grid_shape


class FakeNes:
    write_axis_limits = {
        "t_min": None,
        "t_max": None,
        "z_min": None,
        "z_max": None,
        "y_min": 0,
        "y_max": 12,
        "x_min": None,
        "x_max": None,
    }
    lev = {"data": np.zeros((1,))}
    lat = {"data": np.zeros((12, 13))}
    lon = {"data": np.zeros((12, 13))}
    lat_bnds = {"data": np.zeros((12, 13, 4))}
    lon_bnds = {"data": np.zeros((12, 13, 4))}
    variables = {
        "NOx": {
            "units": "kg.h-1",
            "long_name": "Nitrogen oxides",
            "data": np.zeros((1, 12, 13)),
        },
        "CO": {
            "units": "kg.h-1",
            "data": np.zeros((1, 12, 13)),
        },
    }
    global_attrs = {
        "Conventions": "CF-1.7",
    }

    @staticmethod
    def get_full_shape():
        return 12, 13


class TestAuxiliaryConsistency(unittest.TestCase):
    def test_check_grid_passes_expected_shape(self):
        self.assertTrue(check_grid(FakeNes(), expected_shape=(12, 13)))

    def test_check_grid_geostructure_passes_expected_grid(self):
        grid_gdf = self.create_grid_geostructure()

        self.assertTrue(
            check_grid_geostructure(
                grid_gdf,
                expected_shape=(2, 2),
                expected_geostructure=grid_gdf.copy(),
            )
        )

    def test_check_grid_geostructure_fails_on_cell_count(self):
        grid_gdf = self.create_grid_geostructure(n_cells=3)

        self.assertFalse(check_grid_geostructure(grid_gdf, expected_shape=(2, 2)))

    def test_check_grid_geostructure_fails_on_bounds(self):
        grid_gdf = self.create_grid_geostructure()
        bad_grid_gdf = self.create_grid_geostructure(x_offset=10)

        self.assertFalse(
            check_grid_geostructure(
                bad_grid_gdf,
                expected_shape=(2, 2),
                expected_geostructure=grid_gdf,
            )
        )

    def test_check_aux_file_passes_requested_checks(self):
        self.assertTrue(
            check_aux_file(
                FakeNes(),
                nes_object_to_compare=FakeNes(),
                var_list=["NOx", "CO"],
                global_attributes=["Conventions"],
                variable_attributes={
                    "NOx": ["units", "long_name"],
                },
            )
        )

    def test_check_aux_file_skips_not_requested_checks(self):
        self.assertTrue(check_aux_file(FakeNes(), var_list=["NOx"]))

    def test_check_aux_file_fails_on_missing_variable_attribute(self):
        variable_attributes = {
            "CO": ["long_name"],
        }

        self.assertFalse(check_aux_file(FakeNes(), variable_attributes=variable_attributes))

    def test_check_aux_file_checks_all_var_list_variable_attributes(self):
        self.assertTrue(check_aux_file(FakeNes(), var_list=["NOx"], variable_attributes=["units"]))

    def test_check_aux_file_fails_on_local_shape(self):
        class BadAuxiliary(FakeNes):
            lat = {"data": np.zeros((11, 13))}

        self.assertFalse(check_aux_file(BadAuxiliary(), nes_object_to_compare=FakeNes()))

    def test_get_expected_grid_shape_global(self):
        options = SimpleNamespace(domain_type="global", inc_lat=1.0, inc_lon=1.40625)

        self.assertEqual(get_expected_grid_shape(options), (180, 256))

    def test_get_expected_grid_shape_lcc(self):
        options = SimpleNamespace(domain_type="lcc", ny=100, nx=200)

        self.assertEqual(get_expected_grid_shape(options), (100, 200))

    @staticmethod
    def create_grid_geostructure(x_offset=0, n_cells=4):
        return FakeGeoStructure(
            fids=list(range(n_cells)),
            bounds=(x_offset, 0, x_offset + 2, 2),
        )


class FakeGeoStructure:
    columns = ["FID"]
    crs = "EPSG:4326"
    geometry = None
    index = SimpleNamespace(name=None)

    def __init__(self, fids, bounds):
        self.fids = fids
        self.total_bounds = bounds
        self.geometry = FakeGeometry()

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, item):
        if item == "FID":
            return self.fids
        raise KeyError(item)

    def copy(self):
        return FakeGeoStructure(self.fids.copy(), self.total_bounds)


class FakeGeometry:
    is_empty = None

    def __init__(self):
        self.is_empty = FakeBoolSeries(False)

    @staticmethod
    def isna():
        return FakeBoolSeries(False)


class FakeBoolSeries:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self.value


if __name__ == "__main__":
    unittest.main()
