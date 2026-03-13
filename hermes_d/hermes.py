#!/usr/bin/env python

# Copyright 2018 Earth Sciences Department, BSC-CNS
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

import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from geopandas import GeoDataFrame
import hermes_d
from hermes_d.config import (
    ConfigDelta,
    configure_logger,
    finalize_time_log,
    get_time_stamp,
    log_message,
    precision,
    record_time,
)

from hermes_d.modules import Sector, select_clip, select_grid
from hermes_d.modules.speciation import speciate
from hermes_d.modules.temporal import calculate_date_array
from hermes_d.modules.vertical import (
    distribute_point_sources_vertically,
    distribute_seie_vertically,
)
from mpi4py import MPI
from nes import Nes

# from memory_profiler import profile
global full_time


class HermesDelta(object):
    """
    Interface of the HERMESv3_Delta

    Attributes
    ----------
    comm : MPI.Comm
        MPI Communicator
    config : ConfigDelta
        Configuration object
    options : NameSpace
        Configuration options
    only_horizontal : bool
        Indicates if you want to do only the horizontal distribution.
    first_time : bool
        Indicates if you want only to create the auxiliary files
    grid : Nes
        Grid
    clip : GeoDataFrame
        Clip
    sector_list : list of Sector
        List of sectors to compute
    out_format : str
        Output format type.
    time_step_list : list of datetime
        List of time-steps to simulate

    """

    def __init__(self, config: ConfigDelta, new_date: Optional[datetime] = None, comm: Optional[MPI.Comm] = None):
        """
        Parameters
        ----------
        config : ConfigDelta
            Configuration object
        new_date : datetime or None
            New datetime to simulate.
        comm : MPI.Comm or None
            MPI communicator
        """
        # Initializing general HERMES attributes
        global full_time
        st_time = full_time = get_time_stamp()

        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.master = self.rank == 0

        self.config = config
        self.options = config.options
        self.only_horizontal = self.options.only_horizontal

        # Updating starting date
        if new_date is not None:
            self.start_date = new_date
        else:
            self.start_date = self.options.start_date

        # Logger
        configure_logger(log_path=str(os.path.join(
            self.options.output_dir, "logs", os.path.basename(self.config.get_output_name(self.start_date)))).replace(
            ".nc", "_<rank>.log"), log_detail=self.options.log_level)

        self.first_time = self.config.options.first_time

        log_message("===== Starting HERMESv3 initialization =====")

        if self.options.output_model in ["CMAQ", "WRF_CHEM"] and self.options.domain_type == "global":
            if self.master:
                raise AttributeError("ERROR: Global domain is not available for {0} output model.".format(
                    self.options.output_model))
            sys.exit(1)

        # Initializing HERMES Grid
        self.grid = select_grid(self.comm, self.options)
        if self.only_horizontal:
            self.grid.set_levels({"data": np.array([0]), "units": "-"})

        # Initializing HERMES Clip
        self.clip = select_clip(self.grid, self.options.clipping)

        # Get the time step list
        if self.only_horizontal:
            self.time_step_list = [self.start_date]
        else:
            self.time_step_list = calculate_date_array(
                self.options.start_date,
                self.options.output_timestep_type,
                self.options.output_timestep_num,
                self.options.output_timestep_freq,
            )
        log_message("self.time_step_list: {0}".format(self.time_step_list), level=5)

        self.out_format = self.options.output_model
        self.sector_list = self.make_sector_list()

        # # Precalculate the time information for the grid
        # self.grid = add_time_info_to_grid(self.time_step_list, self.grid)

        log_message("===== End of HERMESv3 initialization =====")
        record_time("Init", "TOTAL", get_time_stamp() - st_time)

    def make_sector_list(self) -> List[Sector]:
        """
        Create the list of sectors to compute.

        Returns
        -------
        List[Sector]
        """
        st_time = get_time_stamp()
        log_message("Creating Sectors", level=1)

        sector_list = []
        for sector_name, sector_info in self.options.sector_information["SECTORS"].items():
            # Adding common paths to proxies
            sector_info["Proxies"]["NUTS2_shp"] = self.options.nuts2_shapefile
            sector_info["Proxies"]["Proxy_aux_path"] = os.path.join(
                self.options.auxiliary_files_path, sector_name, "{0}_proxy.nc".format(sector_name))

            # Add common speciation file to "Profiles"
            sector_info["Profiles"]["individual_speciation_file"] = self.options.individual_speciation_file
            sector_info["Profiles"]["spec_aux_path"] = os.path.join(
                self.options.auxiliary_files_path, sector_name, "{0}_speciation_mapping.txt".format(sector_name))
            # Add nuts2_info to "Profiles"
            sector_info["Profiles"]["nuts2_info"] = self.options.nuts2_info

            input_pollutants = sector_info["Pollutants"] if "Pollutants" in sector_info.keys() else None

            sector = Sector(
                sector_name=sector_name,
                grid=self.grid,
                clip=self.clip,
                only_horizontal=self.only_horizontal,
                data_paths=sector_info["Data"],
                proxies_paths=sector_info["Proxies"],
                profiles_paths=sector_info["Profiles"],
                pollutants=input_pollutants,
                time_step_list=self.time_step_list,
                holidays=self.options.holidays,
                consistency_sei=self.options.consistency_SEI,
                mass_emissions=self.out_format in ['DEFAULT', 'NES'],
                daily_consistency=self.options.daily_consistency,
            )
            sector_list.append(sector)

        record_time("Init", "make_sector_list", get_time_stamp() - st_time)

        return sector_list

    def prepare_inventories(self) -> None:
        """
        Process all the emission inventories to calculate:
        1. Speciation
        2. Vertical distribution
        3. Temporal distribution
        4. Horizontal distribution
        """

        st_time = get_time_stamp()

        for num, sector in enumerate(self.sector_list):
            log_message("Processing {0} sector. ({1}/{2}):".format(
                sector.name, num + 1, len(self.sector_list)), level=2, )

            # Speciation: Convert original pollutants to chemical mechanism species
            st_time = get_time_stamp()
            if sector.profiles_paths["individual_speciation_file"]:
                log_message("Speciation", level=3)
                sector.sector_info = speciate(
                    sector.sector_info, sector.speciation_mapping, sector.species_list, sector.pollutants)
                if sector.point_sources is not None:
                    sector.point_sources = speciate(
                        sector.point_sources, sector.speciation_mapping, sector.species_list, sector.pollutants)
                # Update pollutants with chemech species
                sector.pollutants = sector.species_list

            record_time(f"{sector.name}", "Speciation", get_time_stamp() - st_time)

            # Vertical: Expand data frame by level
            st_time = get_time_stamp()
            if not sector.only_horizontal:
                log_message("Vertical distribution", level=3)
                sector.sector_info = distribute_seie_vertically(
                    sector.vertical_factors, sector.sector_info, sector.grid.lev["data"], sector.pollutants)
                if sector.point_sources is not None:
                    sector.point_sources = distribute_point_sources_vertically(
                        sector.point_sources, sector.grid.lev["data"])
            else:
                # No vertical distribution, all at surface
                sector.sector_info["level"] = 0
                sector.sector_info = sector.sector_info.drop(columns="P_vert")
                if sector.point_sources is not None:
                    sector.point_sources["level"] = 0
                    sector.point_sources = sector.point_sources.drop(columns=["height", "plume_rise_factor"])
            record_time(f"{sector.name}", "Vertical", get_time_stamp() - st_time)

            st_time = get_time_stamp()
            sector.update_sector_info()
            # NOTE: This function must be run after the update_sector_info(). Why?
            if sector.point_sources is not None:
                log_message("Allocating point sources", level=3)
                sector.point_sources = sector.allocate_point_sources()
            record_time(f"{sector.name}", "PointSourcesAllocation", get_time_stamp() - st_time)

        record_time("EmissionInventory", "prepare_inventories", get_time_stamp() - st_time)
        return None

    def prepare_output(self) -> Nes:
        """
        Prepares the output to be written

        Returns
        -------
        Nes
            Output Nes object with lazy variables
        """
        st_time = get_time_stamp()

        result = self.grid.copy(copy_vars=False)
        result.cell_measures = self.grid.cell_measures
        result.set_strlen(None)
        result.set_communicator(self.comm)
        if self.only_horizontal:
            result.set_levels({"data": np.array([0], dtype=precision), "units": "m", "positive": "up"})
        else:
            result.set_levels(
                {"data": np.array(self.options.vertical_description, dtype=precision), "units": "m", "positive": "up"})
        result.set_time(self.time_step_list)
        result.variables = self.sector_list[0].prepare_lazy_output_emissions()

        # Global attributes
        result.global_attrs = {}
        if self.out_format == "CMAQ":
            if self.options.output_attributes is not None:
                global_attributes = pd.read_csv(self.options.output_attributes, sep=",", index_col="attribute")
                for attr_name in [
                    "EXEC_ID",
                    "FTYPE",
                    "NTHIK",
                    "VGTYP",
                    "VGTOP",
                    "VGLVLS",
                    "GDNAM",
                ]:
                    if attr_name in global_attributes.index:
                        result.global_attrs[attr_name] = global_attributes.loc[attr_name, "value"]

            result.global_attrs["FILEDESC"] = "Emissions generated by HERMESv3_Delta."
        elif self.out_format == "WRF_CHEM":
            if self.options.output_attributes is not None:
                global_attributes = pd.read_csv(self.options.output_attributes, sep=",", index_col="attribute")
                for attr_name in [
                    "DAMPCOEF",
                    "KHDIF",
                    "KVDIF",
                    "CEN_LAT",
                    "CEN_LON",
                    "DT",
                    "BOTTOM-TOP_GRID_DIMENSION",
                    "DIFF_OPT",
                    "KM_OPT",
                    "DAMP_OPT",
                    "MP_PHYSICS",
                    "RA_LW_PHYSICS",
                    "RA_SW_PHYSICS",
                    "SF_SFCLAY_PHYSICS",
                    "SF_SURFACE_PHYSICS",
                    "BL_PBL_PHYSICS",
                    "CU_PHYSICS",
                    "SF_LAKE_PHYSICS",
                    "SURFACE_INPUT_SOURCE",
                    "SST_UPDATE",
                    "GRID_FDDA",
                    "GFDDA_INTERVAL_M",
                    "GFDDA_END_H",
                    "GRID_SFDDA",
                    "SGFDDA_INTERVAL_M",
                    "SGFDDA_END_H",
                    "BOTTOM-TOP_PATCH_START_UNSTAG",
                    "BOTTOM-TOP_PATCH_END_UNSTAG",
                    "BOTTOM-TOP_PATCH_START_STAG",
                    "BOTTOM-TOP_PATCH_END_STAG",
                    "GRID_ID",
                    "PARENT_ID",
                    "I_PARENT_START",
                    "J_PARENT_START",
                    "PARENT_GRID_RATIO",
                    "NUM_LAND_CAT",
                    "ISWATER",
                    "ISLAKE",
                    "ISICE",
                    "ISURBAN",
                    "ISOILWATER",
                    "GRIDTYPE",
                    "MMINLU",
                ]:
                    if attr_name in global_attributes.index:
                        result.global_attrs[attr_name] = global_attributes.loc[attr_name, "value"]
        elif self.out_format == "CHIMERE":
            if self.options.output_attributes is not None:
                global_attributes = pd.read_csv(self.options.output_attributes, sep=",", index_col="attribute")
                for attr_name in [
                    "Title",
                    "Sub-title",
                    "Generating_process",
                    "Conventions",
                    "Domain",
                ]:
                    if attr_name in global_attributes.index:
                        result.global_attrs[attr_name] = global_attributes.loc[attr_name, "value"]

        if self.out_format != "CHIMERE":
            history_attr = "HISTORY"
        else:
            history_attr = "history"

        result.global_attrs[history_attr] = (
            "Code developed by Barcelona Supercomputing Center (BSC, https://www.bsc.es/). "
            + "Developer: Carles Tena Medina (carles.tena@bsc.es) "
        )

        record_time(f"Write", "Creation", get_time_stamp() - st_time)
        return result

    def get_time_step_emissions(
        self, time_step: datetime, i_time: int
    ) -> Dict[str, Any]:
        """
        Obtain the emissions of the selected time step

        Parameters
        ----------
        time_step : datetime
            Time stamp to simulate
        i_time : int
            Index of the time-step

        Returns
        -------
        Dict[str, Any]
            Time stamp emissions in Nes format
        """
        result = None
        for num, sector in enumerate(self.sector_list):
            if result is None:
                result = sector.get_4d_emissions(i_time)
            else:
                for var_name, var_info in sector.get_4d_emissions(i_time).items():
                    result[var_name]["data"] += var_info["data"]
        return result

    def main(self) -> Optional[datetime]:
        """
        Main functionality of the model.

        Returns
        -------
        datetime
            New datetime if other simulation is requested, otherwise None
        """
        if self.first_time:
            # Stop run
            log_message("***** HERMESv3_Delta First Time finished successfully *****")

        else:
            write_time = 0
            st_time = get_time_stamp()
            log_message("")
            log_message("***** Starting HERMESv3_Delta *****")

            log_message("Preparing sectors", level=1)

            self.prepare_inventories()

            # Preparing output file
            aux_time = get_time_stamp()
            result = self.prepare_output()
            out_path = self.config.get_output_name(self.start_date)
            log_message("Creating output file", level=1)
            result.to_netcdf(
                out_path,
                serial=self.options.serial_write,
                keep_open=True,
                nc_type=self.out_format,
            )
            write_time += get_time_stamp() - aux_time
            log_message("Empty output file created", level=2)
            log_message("{0}".format(out_path), level=3)

            for i_time, time_step in enumerate(self.time_step_list):
                log_message("Creating {0} data ({1}/{2})".format(
                    time_step, i_time + 1, len(self.time_step_list)), level=2,)
                result.variables = self.get_time_step_emissions(time_step, i_time)
                log_message("Filling data", level=3)
                aux_time = get_time_stamp()
                result.append_time_step_data(i_time, out_format=self.out_format)
                write_time += get_time_stamp() - aux_time
            # Closing output file
            aux_time = get_time_stamp()
            result.close()
            write_time += get_time_stamp() - aux_time

            # Finishing
            log_message("***** HERMESv3_Delta simulation finished successfully *****")

            record_time("Calcul", "TOTAL", get_time_stamp() - st_time - write_time)
            record_time("Write", "TOTAL", write_time)
        record_time("HERMES", "TOTAL", get_time_stamp() - full_time)

        finalize_time_log(str(os.path.join(self.options.output_dir, "logs", os.path.basename(
            self.config.get_output_name(self.start_date)),)).replace(
            ".nc", f"_Times_{str(self.size).zfill(4)}.csv"))

        # Updating the date for a date-loop HERMES simulation
        if self.start_date < self.options.end_date:
            if self.options.output_timestep_type in ["hourly", "daily"]:
                new_date = self.start_date + timedelta(days=1)
            elif self.options.output_timestep_type in ["monthly"]:
                new_date = self.start_date + relativedelta(months=1)
            elif self.options.output_timestep_type in ["yearly"]:
                new_date = self.start_date + relativedelta(years=1)
            else:
                raise RuntimeError("Unknown timestep_type")
            return new_date

        return None


def run() -> None:
    """
    Run the HERMESv3_Delta model.

    This function initializes and runs the HERMESv3_Delta model using the provided configuration.
    The model continues to simulate emissions for subsequent time steps until the end date is reached.

    Returns
    -------
    None
    """

    try:
        config = ConfigDelta()
        if hermes_d.DEBUG:
            from pandas import set_option
            set_option("display.max_columns", None)

        model = HermesDelta(config)

        date = model.main()
        while date is not None:
            date = HermesDelta(config, new_date=date).main()
    except Exception as e:
        log_message(f"Process {MPI.COMM_WORLD.Get_rank()}: Critical error detected {e}, aborting MPI.", level=9)
        log_message(f"Process {MPI.COMM_WORLD.Get_rank()}: Traceback:\n{traceback.format_exc()}", level=9,)

        MPI.COMM_WORLD.Abort(1)

    return


if __name__ == "__main__":
    run()
