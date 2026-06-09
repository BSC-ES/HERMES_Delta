#!/usr/bin/env python
import unittest
import shutil
import pytest
import tempfile
import csv
import os
import pandas as pd

from hermes_d.modules.speciation import SpeciationDelta, speciate
from hermes_d.config import configure_logger

configure_logger(log_path=str("logs_unittests/log_speciation_testing.log"))


class TestSpeciationDelta(unittest.TestCase):
    def setUp(self):
        self.speciation = SpeciationDelta(None, None, None, None, None)

    # =============================================
    # 1. get_speciation_mapping
    # =============================================

    def test_get_speciation_mapping(self):
        spec_snap = [
            [
                "P_spec",
                "CAS",
                "Species_name",
                "Profile_code",
                "SNAP_activity",
                "TOG_to_VOC_ratio",
                "Weight_percentual",
                "MWt",
                "Specie_ori",
            ],
            [
                "E_050501",
                "pm25_other",
                "pm25_other",
                None,
                "050501",
                1,
                54.8,
                1,
                "pm25",
            ],
            [
                "E_050102",
                "10024-97-2",
                "nitrous oxide",
                None,
                "050102",
                1,
                100,
                44.01,
                "n2o",
            ],
            [
                "E_050201",
                "10024-97-2",
                "nitrous oxide",
                None,
                "050201",
                1,
                100,
                44.01,
                "n2o",
            ],
        ]

        indi_spec = [
            [
                "CAS",
                "Species_name",
                "EMNO",
                "EMNO_2",
                "EMCO",
                "EMSO_2",
                "EMNH_3",
                "EMHCHO",
                "EMETH",
                "EMHC3",
                "EMHC5",
                "EMHC8",
                "EMETE",
                "EMOLT",
                "EMDIEN",
                "EMISO",
                "EMKET",
                "EMAPI",
                "EMLIM",
                "EMTOL",
                "EMXYL",
                "EMCH_3CCl_3",
                "EMDMS",
                "EMHCl",
                "EMN_2O",
                "EMCH_4",
                "EMALD",
                "EMCSL",
                "EMBLACKC1",
                "EMBLACKC2",
                "EMBLACKC3",
                "EMBLACKC4",
                "EMBLACKC5",
                "EMBLACKC6",
                "EMORGANC1",
                "EMORGANC2",
                "EMORGANC3",
                "EMORGANC4",
                "EMORGANC5",
                "EMORGANC6",
                "EMSORGAN1",
                "EMSORGAN2",
                "EMSORGAN3",
                "EMSORGAN4",
                "EMSORGAN5",
                "EMSORGAN6",
            ],
            [
                "units",
                "",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "mol/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
                "kg/m2/s",
            ],
            [
                "short_description",
                "",
                "nitrogen_monoxide",
                "nitrogen_dioxide",
                "carbon_monoxide",
                "sulfur_dioxide",
                "ammonia",
                "formaldehyde",
                "ethane",
                "alkanes_alcohols_esters_alkynes_HO_rate_2.9_constant",
                "alkanes_alcohols_esters_alkynes_HO_rate_4.8_constant",
                "alkanes_alcohols_esters_alkynes_HO_rate_7.9_constant",
                "ethene",
                "terminal_alkenes",
                "butadiene_other_dienes",
                "isoprene",
                "ketones",
                "alpha_pinene_other_cyclic_terpenes",
                "d-limonene",
                "toluene_less_reactive_aromatics",
                "xylenes_more_reactive_aromatics",
                "methylchloroform",
                "dimethylsulphide",
                "hydrogen_chloride",
                "nitrous_oxide",
                "methane",
                "acetaldehyde_higher_aldehydes",
                "Cresol_other_hydroxy_substituted_aromatics",
                "black_carbon_bin1",
                "black_carbon_bin2",
                "black_carbon_bin3",
                "black_carbon_bin4",
                "black_carbon_bin5",
                "black_carbon_bin6",
                "organic_carbon_bin1",
                "organic_carbon_bin2",
                "organic_carbon_bin3",
                "organic_carbon_bin4",
                "organic_carbon_bin5",
                "organic_carbon_bin6",
                "secondari_organic_carbon_bin1",
                "secondari_organic_carbon_bin2",
                "secondari_organic_carbon_bin3",
                "secondari_organic_carbon_bin4",
                "secondari_organic_carbon_bin5",
                "secondari_organic_carbon_bin6",
            ],
            [
                "pm25_other",
                "pm25_other",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
            ],
            [
                "10024-97-2",
                "nitrous oxide",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "1",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
            ],
        ]

        # First CSV
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", mode="w", newline=""
        ) as tmp_spec:
            spec_snap_path = tmp_spec.name
            writer = csv.writer(tmp_spec)
            writer.writerows(spec_snap)

        # Second CSV
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", mode="w", newline=""
        ) as tmp_indi:
            indi_spec_path = tmp_indi.name
            writer = csv.writer(tmp_indi)
            writer.writerows(indi_spec)

            self.speciation.spec_snap_path = spec_snap_path
            self.speciation.indi_spec_path = indi_spec_path

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

        duplicates, mapping, species_list = self.speciation.get_speciation_mapping(
            pollutants
        )

        assert set(duplicates.keys()) == {"E_0001", "E_0002"}
        assert duplicates["E_0002"]["EMN_2O"]["n2o"] == pytest.approx(
            22.722108611679165, abs=1e-10
        )
        assert mapping == {
            "E_050501": "E_0001",
            "E_050102": "E_0002",
            "E_050201": "E_0002",
        }

        assert species_list == [
            "EMNO",
            "EMNO_2",
            "EMCO",
            "EMSO_2",
            "EMNH_3",
            "EMHCHO",
            "EMETH",
            "EMHC3",
            "EMHC5",
            "EMHC8",
            "EMETE",
            "EMOLT",
            "EMDIEN",
            "EMISO",
            "EMKET",
            "EMAPI",
            "EMLIM",
            "EMTOL",
            "EMXYL",
            "EMCH_3CCl_3",
            "EMDMS",
            "EMHCl",
            "EMN_2O",
            "EMCH_4",
            "EMALD",
            "EMCSL",
            "EMBLACKC1",
            "EMBLACKC2",
            "EMBLACKC3",
            "EMBLACKC4",
            "EMBLACKC5",
            "EMBLACKC6",
            "EMORGANC1",
            "EMORGANC2",
            "EMORGANC3",
            "EMORGANC4",
            "EMORGANC5",
            "EMORGANC6",
            "EMSORGAN1",
            "EMSORGAN2",
            "EMSORGAN3",
            "EMSORGAN4",
            "EMSORGAN5",
            "EMSORGAN6",
        ]

    # =============================================
    # 2. map_emission_p_spec
    # =============================================
    def test_map_emission_p_spec(self):
        emissions = pd.DataFrame(
            {
                "P_spec": ["E_050102", "E_050401", "E_050401"],
                "is_point_source": [False, False, False],
            }
        )

        mapping = {
            "E_050501": "E_0001",
            "E_050401": "E_0002",
            "E_050503": "E_0003",
            "E_050502": "E_0001",
            "E_050201": "E_0004",
            "E_050303": "E_0005",
            "E_050601": "E_0006",
            "E_050202": "E_0004",
            "E_050603": "E_0007",
            "E_050302": "E_0005",
            "E_050102": "E_0008",
        }
        sector_name = "SNAP05"
        emissions = self.speciation.map_emission_p_spec(emissions, mapping, sector_name)
        assert emissions["P_spec"].tolist() == ["E_0008", "E_0002", "E_0002"]

    # =============================================
    # 3. speciate
    # =============================================

    def test_speciate(self):
        emissions = pd.DataFrame(
            {
                "is_point_source": {("050102", "ES12"): False},
                "spatial_proxy": {("050102", "ES12"): "shapefile_polygon"},
                "proxy_code": {("050102", "ES12"): "coal_mining"},
                "P_month": {("050102", "ES12"): None},
                "P_hour": {("050102", "ES12"): None},
                "P_week": {("050102", "ES12"): None},
                "P_day": {("050102", "ES12"): None},
                "P_spec": {("050102", "ES12"): "E_0008"},
                "P_vert": {("050102", "ES12"): "V_050102_ES12"},
                "nmvoc": {("050102", "ES12"): 0.0},
                "ch4": {("050102", "ES12"): 949.676},
            }
        )

        speciation_factors = {
            "E_0001": {
                "EMHC8": {"nmvoc": 0.001466625705733058},
                "EMCH_4": {"ch4": 0.06234413965087282},
            },
            "E_0002": {
                "EMHC8": {"nmvoc": 0.0012356962301311432},
                "EMCH_4": {"ch4": 0.06234413965087282},
            },
            "E_0003": {
                "EMHC8": {"nmvoc": 0.0005843932247110425},
                "EMCH_4": {"ch4": 0.06234413965087282},
            },
            "E_0004": {
                "EMHC8": {"nmvoc": 0.0036572161862083808},
                "EMCH_4": {"ch4": 0.06234413965087282},
            },
            "E_0005": {"EMCH_4": {"ch4": 0.06234413965087282}},
            "E_0006": {
                "EMHC8": {"nmvoc": 0.0056534633214023475},
                "EMCH_4": {"ch4": 0.06234413965087282},
            },
            "E_0007": {
                "EMHC8": {"nmvoc": 0.0004064019107429055},
                "EMCH_4": {"ch4": 0.06234413965087282},
            },
            "E_0008": {"EMCH_4": {"ch4": 0.06234413965087282}},
        }

        species_list = ["EMCH_4", "EMHC8"]
        original_pollutants = ["nmvoc", "ch4"]

        emissions = speciate(
            emissions, speciation_factors, species_list, original_pollutants
        )

        assert set(emissions.columns) == set(
            [
                "is_point_source",
                "spatial_proxy",
                "proxy_code",
                "P_month",
                "P_hour",
                "P_week",
                "P_day",
                "P_vert",
                "EMHC8",
                "EMCH_4",
            ]
        )
        assert set(emissions["EMCH_4"][emissions["EMCH_4"] != 0]) == pytest.approx(
            [59.2067331670823], abs=1e-10
        )
        assert set(emissions["EMHC8"][emissions["EMHC8"] != 0]) == set()

    @classmethod
    def tearDownClass(cls):
        logfolder = "logs_unittests"
        if os.path.exists(logfolder):
            shutil.rmtree(logfolder)


if __name__ == "__main__":
    unittest.main()
