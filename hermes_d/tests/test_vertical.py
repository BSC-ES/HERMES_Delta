#!/usr/bin/env python
import unittest
import yaml
import numpy as np
import pandas as pd
import tempfile
import shutil
import os

from hermes_d.modules.vertical import VerticalDelta, distribute_seie_vertically, distribute_point_sources_vertically
from hermes_d.config import configure_logger

configure_logger(log_path=str("logs_unittests/log_vertical_testing.log"))


class TestVerticalDelta(unittest.TestCase):
    def setUp(self):
        self.vertical = VerticalDelta(None, None, None)

    def remove_yaml(self, yaml_path):
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

    # =============================================
    # 1. get_vertical_profiles
    # =============================================

    def test_01_a(self):
        sector_info = pd.DataFrame(
            {"P_vert": ["V_050101_ES11", "V_050101_ES12", "V_050102_ES12"]}
        )
        sector_info.index = pd.MultiIndex.from_tuples(
            [("050101", "ES11"), ("050101", "ES12"), ("050102", "ES12")]
        )

        vertical_profiles = {
            "Vertical_profiles": {
                "V_050101_ES11": {20: 1.0},
                "V_050101_ES12": {10: 0.2, 50: 0.3, 200: 0.5},
                "V_050102_ES12": {20: 0.1, 50: 0.4, 100: 0.5},
                "V_050103_ES13": {50: 0, 100: 1.0},
                "V_050104_ES14": {0: 1.0},
                "V_050105_ES15": {100: 1.0},
                "V_050106_ES15": {100: 1.0},
                "V_050106_ES16": {
                    100: 0,
                    200: 0.1,
                    300: 0.05,
                    400: 0.05,
                    500: 0.2,
                    600: 0.2,
                    700: 0.2,
                    800: 0.14,
                    900: 0.05,
                    1000: 0.01,
                },
            }
        }

        # file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        vertical_profiles_path = tmp.name
        with open(vertical_profiles_path, "w") as f:
            yaml.dump(vertical_profiles, f)

        output_layers = [10, 50, 100, 500]

        self.vertical.profiles_paths = {"Vertical_profiles": vertical_profiles_path}
        self.vertical.sector_info = sector_info
        self.vertical.output_layers = output_layers

        res2, res3 = self.vertical.get_vertical_profiles()
        assert res2 == {
            "V_0001": {20: 1.0},
            "V_0002": {10: 0.2, 50: 0.3, 200: 0.5},
            "V_0003": {20: 0.1, 50: 0.4, 100: 0.5},
        }
        assert res3 == {
            "V_050101_ES11": "V_0001",
            "V_050101_ES12": "V_0002",
            "V_050102_ES12": "V_0003",
        }

        self.remove_yaml(vertical_profiles_path)

    def test_01_b(self):
        sector_info = pd.DataFrame(
            {"P_vert": ["V_050101_ES11", "V_050101_ES12", "V_050102_ES12"]}
        )
        sector_info.index = pd.MultiIndex.from_tuples(
            [("050101", "ES11"), ("050101", "ES12"), ("050102", "ES12")]
        )

        sector_info.loc[("050101", "ES11"), "P_vert"] = np.nan
        vertical_profiles = {
            "Vertical_profiles": {
                "V_050101_ES11": {20: 1.0},
                "V_050101_ES12": {10: 0.2, 50: 0.3, 200: 0.5},
                "V_050102_ES12": {20: 0.1, 50: 0.4, 100: 0.5},
                "V_050103_ES13": {50: 0, 100: 1.0},
                "V_050104_ES14": {0: 1.0},
                "V_050105_ES15": {100: 1.0},
                "V_050106_ES15": {100: 1.0},
                "V_050106_ES16": {
                    100: 0,
                    200: 0.1,
                    300: 0.05,
                    400: 0.05,
                    500: 0.2,
                    600: 0.2,
                    700: 0.2,
                    800: 0.14,
                    900: 0.05,
                    1000: 0.01,
                },
            }
        }

        # file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        vertical_profiles_path = tmp.name
        with open(vertical_profiles_path, "w") as f:
            yaml.dump(vertical_profiles, f)

        output_layers = [10, 50, 100, 500]

        self.vertical.profiles_paths = {"Vertical_profiles": vertical_profiles_path}
        self.vertical.sector_info = sector_info
        self.vertical.output_layers = output_layers

        res1, res2 = self.vertical.get_vertical_profiles()

        assert res1 == {
            "V_0001": {0: 1.0},
            "V_0002": {10: 0.2, 50: 0.3, 200: 0.5},
            "V_0003": {20: 0.1, 50: 0.4, 100: 0.5},
        }
        assert res2 == {
            "surface": "V_0001",
            "V_050101_ES12": "V_0002",
            "V_050102_ES12": "V_0003",
        }

        self.remove_yaml(vertical_profiles_path)

    # =============================================
    # 2. map_sector_info_p_vert
    # =============================================

    def test_02_a(self):
        sector_info = pd.DataFrame(
            {"P_vert": ["V_050101_ES11", "V_050101_ES12", "V_050102_ES12"]}
        )
        sector_info.index = pd.MultiIndex.from_tuples(
            [("050101", "ES11"), ("050101", "ES12"), ("050102", "ES12")]
        )

        dict_map = {
            "V_050101_ES11": "V_0001",
            "V_050101_ES12": "V_0002",
            "V_050102_ES12": "V_0003",
        }
        self.vertical.sector_info = sector_info
        res = self.vertical.map_sector_info_p_vert(dict_map)
        assert res["P_vert"].values.tolist() == ["V_0001", "V_0002", "V_0003"]

    def test_02_b(self):
        sector_info = pd.DataFrame(
            {"P_vert": ["V_050101_ES11", "V_050101_ES12", "V_050102_ES12"]}
        )
        sector_info.index = pd.MultiIndex.from_tuples(
            [("050101", "ES11"), ("050101", "ES12"), ("050102", "ES12")]
        )

        dict_map = {
            "V_050101_ES11": "V_0001",
            "V_050101_ES12": "V_0002",
            "V_050102_ES12": "V_0001",
        }
        self.vertical.sector_info = sector_info
        res = self.vertical.map_sector_info_p_vert(dict_map)
        assert res["P_vert"].values.tolist() == ["V_0001", "V_0002", "V_0001"]

    # =============================================
    # 3. calculate_vertical_factors
    # =============================================

    def test_03_a(self):
        vertical_profiles = {
            "V_0001": {20: 1.0},
            "V_0002": {10: 0.2, 50: 0.3, 200: 0.5},
            "V_0003": {20: 0.1, 50: 0.4, 100: 0.5},
        }
        output_layers = [10, 50, 100, 500]
        self.vertical.output_layers = output_layers
        res = self.vertical.calculate_vertical_factors(vertical_profiles)
        assert (res["V_0001"] == np.array([0.5, 0.5, 0, 0])).all()
        tmp = [round(num, 3) for num in res["V_0002"]]
        assert tmp == [0.2, 0.3, 0.167, 0.333]
        assert (res["V_0003"] == np.array([0.05, 0.45, 0.5, 0.0])).all()

    def test_03_b(self):
        vertical_profiles = {"V_0001": {0: 1.0}}
        output_layers = [10, 50, 100, 500]
        self.vertical.output_layers = output_layers
        res = self.vertical.calculate_vertical_factors(vertical_profiles)
        assert (res["V_0001"] == np.array([1, 0, 0, 0])).all()

    # =============================================
    # 4. distribute_seie_vertically
    # =============================================
    def test_04_a(self):
        sector_info = pd.DataFrame(
            {
                "P_vert": ["V_050101_ES11", "V_050101_ES12", "V_050102_ES12"],
                "is_point_source": [False, True, False],
                "spatial_proxy": [
                    "shapefile_polygon",
                    "shapefile_polygon",
                    "shapefile_polygon",
                ],
                "proxy_code": ["population", "population", "population"],
                "P_hour": [None, None, None],
                "P_week": [None, None, None],
                "P_day": [None, None, None],
                "P_month": [None, None, None],
                "P_spec": [None, None, None],
                "nox_no2": [100, 200, 300],
                "nmvoc": [100, 200, 300],
                "sox": [100, 200, 300],
                "co": [100, 200, 300],
                "nh3": [100, 200, 300],
                "pm25": [100, 200, 300],
                "pm10": [100, 200, 300],
                "bc": [100, 200, 300],
                "co2": [100, 200, 300],
                "ch4": [100, 200, 300],
                "n2o": [100, 200, 300],
            }
        )

        sector_info.index = pd.MultiIndex.from_tuples(
            [("050101", "ES11"), ("050101", "ES12"), ("050102", "ES12")]
        )
        sector_info.index.names = ["SNAP_activity", "NUTS2_code"]

        vertical_profiles = {
            "Vertical_profiles": {
                "V_050101_ES11": {20: 1.0},
                "V_050101_ES12": {10: 0.2, 50: 0.3, 200: 0.5},
                "V_050102_ES12": {20: 0.1, 50: 0.4, 100: 0.5},
                "V_050103_ES13": {50: 0, 100: 1.0},
                "V_050104_ES14": {0: 1.0},
                "V_050105_ES15": {100: 1.0},
                "V_050106_ES15": {100: 1.0},
                "V_050106_ES16": {
                    100: 0,
                    200: 0.1,
                    300: 0.05,
                    400: 0.05,
                    500: 0.2,
                    600: 0.2,
                    700: 0.2,
                    800: 0.14,
                    900: 0.05,
                    1000: 0.01,
                },
            }
        }

        # file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        vertical_profiles_path = tmp.name
        with open(vertical_profiles_path, "w") as f:
            yaml.dump(vertical_profiles, f)

        output_layers = [10, 50, 100, 500]

        self.vertical.profiles_paths = {"Vertical_profiles": vertical_profiles_path}
        self.vertical.sector_info = sector_info
        self.vertical.output_layers = output_layers

        _, res3 = self.vertical.get_vertical_profiles()
        emissions = self.vertical.map_sector_info_p_vert(res3)
        vertical_factors = {
            "V_0001": np.array([0.5, 0.5, 0, 0]),
            "V_0002": np.array([0.2, 0.3, 0.15, 0.35]),
            "V_0003": np.array([0.05, 0.45, 0.5, 0]),
        }
        pollutants = [
            "nox_no2",
            "nmvoc",
            "sox",
            "co",
            "nh3",
            "pm10",
            "pm25",
            "bc",
            "co2",
            "ch4",
            "n2o",
        ]
        res = distribute_seie_vertically(
            vertical_factors, emissions, output_layers, pollutants
        )
        assert (
            res.columns
            == pd.Index(
                [
                    "is_point_source",
                    "spatial_proxy",
                    "proxy_code",
                    "P_hour",
                    "P_week",
                    "P_day",
                    "P_month",
                    "P_spec",
                    "nox_no2",
                    "nmvoc",
                    "sox",
                    "co",
                    "nh3",
                    "pm25",
                    "pm10",
                    "bc",
                    "co2",
                    "ch4",
                    "n2o",
                    "level",
                ]
            )
        ).all()
        assert res["level"].tolist() == [0, 1, 0, 1, 2, 3, 0, 1, 2]

        self.remove_yaml(vertical_profiles_path)

    # =============================================
    # 5. distribute_point_sources_vertically
    # =============================================

    def test_05_a(self):
        emissions = pd.DataFrame(
            {
                "Code": ["LPS-00516", "LSP-00510"],
                "nox_no2": [100, 100],
                "height": [100, 0],
                "plume_rise_factor": [1.3, 1.3],
                "geometry": [None, None],
            }
        )

        output_layers = [10, 50, 100, 500]
        res = distribute_point_sources_vertically(emissions, output_layers)
        assert sorted(res.columns) == sorted(["Code", "nox_no2", "geometry", "level"])
        assert res["level"].tolist() == [3, 0]

    @classmethod
    def tearDownClass(cls):
        logfolder = "logs_unittests"
        if os.path.exists(logfolder):
            shutil.rmtree(logfolder)


if __name__ == "__main__":
    unittest.main()
