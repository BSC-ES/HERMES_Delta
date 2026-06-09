#!/usr/bin/env python

# Copyright 2023 Earth Sciences Department, BSC-CNS
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
# along with HERMESv3_GR. If not, see <http://www.gnu.org/licenses/>.

from os import path, makedirs
from configargparse import ArgParser, Namespace
from pandas import read_csv
from warnings import warn
from shutil import rmtree
from datetime import datetime
from mpi4py import MPI
from hermes_d import __version__
from yaml import safe_load as load_yml
from hermes_d.utilities import parse_path, parse_list, parse_end_date, parse_bool, parse_start_date, parse_float_list


class ConfigDelta(ArgParser):
    """
    Initialization of the arguments that the parser can handle.

    Attributes
    ----------
    comm : MPI.Comm
        MPI communicator
    options : Namespace
        Configuration options
    """
    def __init__(self, comm=None, config_path=None):
        """
        Initialize the Config class

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        config_path : str
            Path to the configuration file. Only used when launching HERMES as a python library
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        super(ConfigDelta, self).__init__()
        description_csv_file = path.join(path.dirname(path.abspath(__file__)), "arguments_description.csv")
        args_csv = read_csv(description_csv_file, index_col="Parameter", keep_default_na=False, comment="#")
        self.options = self.read_options(args_csv, config_path)
        # Define DEBUG in hermesv3_d according to the chosen log level
        import hermes_d
        hermes_d.DEBUG = self.options.log_level == 5

    # noinspection DuplicatedCode
    def read_options(self, args_csv, config_path=None) -> Namespace:
        """
        Reads all the arguments from command line or from the configuration file.
        The value of an argument given by command line has high priority that the one that appear in the
        configuration file.

        Parameters
        ----------
        args_csv : DataFrame
            Configuration of the arguments to be added to the parser. Includes tha name of each option, its help
            message, data type, 'required' flag, choices option and default value.
        config_path : str
            Path to the configuration file. Only used when launching HERMES as a python library

        Returns
        -------
        Namespace
            Configuration options
        """
        p = ArgParser()
        p.add_argument("-c", "--my-config", required=False, is_config_file=True, default=config_path,
                       help="Path to the configuration file.")

        p.add_argument("--version", "-V", "-v", action="version", version="%(prog)s " + __version__)
        # Process each argument in args_csv
        for arg_name, csv_row in args_csv.iterrows():
            arg_name = arg_name.strip()
            # Parse choices
            if csv_row["Choices"].strip() == "None":
                csv_row["Choices"] = None
            else:
                csv_row["Choices"] = parse_list(csv_row["Choices"])
            # Add to the parser
            p.add_argument(f"--{arg_name}",
                           dest=arg_name,
                           help=csv_row["Description"],
                           type=eval(csv_row["Data Type"]),
                           choices=csv_row["Choices"],
                           required=csv_row["Required"],
                           default=eval(csv_row["Default"]))

        arguments, unknown = p.parse_known_args()
        if len(unknown) > 0:
            warn("Unrecognized arguments: {0}".format(unknown))

        for item in vars(arguments):
            is_str = isinstance(arguments.__dict__[item], str)
            if is_str:
                # Parse paths
                arguments.__dict__[item] = parse_path(path=arguments.__dict__[item],
                                                      data_path=arguments.data_path,
                                                      input_dir=arguments.input_dir,
                                                      version="v" + __version__,
                                                      domain_type=arguments.domain_type)
                # Check domain options and parse <resolution>
                self.process_domain_resolution_arguments(arguments, item)

        if arguments.output_timestep_type != "hourly":
            # This change is to try not take into account daylight saving time errors.
            # at 00:00 depending on the timezone you can be in a place or in others.
            arguments.start_date.replace(hour=12)
        arguments.end_date = parse_end_date(arguments.end_date, arguments.start_date)

        # arguments.first_time = self._parse_bool(arguments.first_time)
        # arguments.erase_auxiliary_files = self._parse_bool(arguments.erase_auxiliary_files)
        # arguments.serial_write = self._parse_bool(arguments.serial_write)
        # arguments.holidays = self._parse_bool(arguments.holidays)

        if arguments.individual_speciation_file == "None":
            arguments.individual_speciation_file = parse_bool(arguments.individual_speciation_file)

        self.create_dir(arguments.output_dir)
        if arguments.erase_auxiliary_files:
            if path.exists(arguments.auxiliary_files_path):
                if self.comm.Get_rank() == 0:
                    rmtree(arguments.auxiliary_files_path)
            self.comm.Barrier()
        self.create_dir(arguments.auxiliary_files_path)

        arguments.sector_information = self.parse_yaml(arguments.sector_information, arguments.data_path,
                                                       arguments.input_dir)

        return arguments

    @staticmethod
    def parse_yaml(file_path, data_path, input_dir):
        """
        Parse the input YAML file that contains the sector description

        Parameters
        ----------
        file_path : str or dict
            Path to the YAML file to be parsed
        data_path : str
            Data path to be used as replacement of "<data_path>"
        input_dir : str
            Input directory path to be used as replacement of "<input_dir>"

        Returns
        -------
        dict
            Dictionary with the parsed YAML information
        """
        def update_dict(dict_aux, old, new):
            """
            Update a dictionary by finding in the dictionary values strings to replace the "old" pattern with the new
            string.

            Parameters
            ----------
            dict_aux : dict
                Dictionary where find the patterns
            old : str
                Pattern to be replaced
            new : str
                String to use as replacement of the pattern

            Returns
            -------
            dict
                Updated dictionary
            """
            for k, v in dict_aux.items():
                if isinstance(v, dict):
                    update_dict(v, old, new)
                elif isinstance(v, str):
                    dict_aux[k] = v.replace(old, new)
            return

        if isinstance(file_path, str):
            with open(file_path, "r") as file:
                conf = load_yml(file)
        else:
            conf = file_path

        # Parse paths
        update_dict(conf, old="<data_path>", new=data_path)
        update_dict(conf, old="<input_dir>", new=input_dir)

        return conf

    def get_output_name(self, date):
        """
        Generates the full path of the output replacing <date> by YYYYMMDDHH, YYYYMMDD, YYYYMM or YYYY depending on the
        output_timestep_type.

        Parameters
        ----------
        date : datetime
            Starting date to simulate

        Returns
        -------
        str
            Complete output path
        """
        if self.options.output_timestep_type == "hourly":
            file_name = self.options.output_name.replace("<date>", date.strftime("%Y%m%d%H"))
        elif self.options.output_timestep_type == "daily":
            file_name = self.options.output_name.replace("<date>", date.strftime("%Y%m%d"))
        elif self.options.output_timestep_type == "monthly":
            file_name = self.options.output_name.replace("<date>", date.strftime("%Y%m"))
        elif self.options.output_timestep_type == "yearly":
            file_name = self.options.output_name.replace("<date>", date.strftime("%Y"))
        else:
            file_name = self.options.output_name

        file_name = file_name.replace("<YYYYMMDDHH>", date.strftime("%Y%m%d%H"))
        file_name = file_name.replace("<YYYYMMDD>", date.strftime("%Y%m%d"))
        file_name = file_name.replace("<YYYYMM>", date.strftime("%Y%m"))
        file_name = file_name.replace("<YYYY>", date.strftime("%Y"))

        full_path = path.join(self.options.output_dir, file_name)
        return full_path

    @staticmethod
    def create_dir(dir_path):
        """
        Create the given folder if it is not created yet.

        Parameters
        ----------
        dir_path : str
            Path to be created

        """
        if not path.exists(dir_path):
            makedirs(dir_path, exist_ok=True)
        return None

    # noinspection DuplicatedCode
    @staticmethod
    def process_domain_resolution_arguments(arguments, item):
        """
        Processes and updates the resolution string in `arguments` based on the `domain_type` and the
        provided resolution increments.

        Parameters
        ----------
        arguments : Namespace
            An object containing various attributes, including `domain_type` and resolution increment
            attributes (e.g.,`domain_type`, `inc_lat`, `inc_lon`, `inc_rlat`, `inc_rlon`, etc.).
        item : str
            The key in the `arguments` dictionary whose value contains the placeholder (`<resolution>`)
            that will be replaced with the actual resolution string.

        Returns
        ------
        None

        Raises
        ------
        RuntimeError
            If the required resolution increment attributes (e.g., `inc_lat`, `inc_lon`, etc.)
            for the specified `domain_type` are not provided.
            If the `domain_type` is unsupported.
        """
        domain_type = arguments.domain_type

        if domain_type in ['global', 'regular']:
            if arguments.inc_lat is not None and arguments.inc_lon is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{arguments.inc_lat}_{arguments.inc_lon}')
            else:
                raise RuntimeError(
                    f"Arguments 'inc_lat' and 'inc_lon' are needed for the specified domain_type '{domain_type}'")

        elif domain_type == 'rotated':
            if arguments.inc_rlat is not None and arguments.inc_rlon is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>',  f'{arguments.inc_rlat}_{arguments.inc_rlon}')
            else:
                raise RuntimeError(
                    f"Arguments 'inc_rlat' and 'inc_rlon' are needed for the specified domain_type '{domain_type}'")

        elif domain_type == 'rotated_nested':
            if arguments.n_rlat is not None and arguments.n_rlon is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{arguments.n_rlat}_{arguments.n_rlon}')
            else:
                raise RuntimeError(
                    f"Arguments 'n_rlat' and 'n_rlon' are needed for the specified domain_type '{domain_type}'")

        elif domain_type in ['lcc', 'mercator']:
            if arguments.inc_x is not None and arguments.inc_y is not None:
                arguments.__dict__[item] = arguments.__dict__[item].replace(
                    '<resolution>', f'{arguments.inc_x}_{arguments.inc_y}')
            else:
                raise RuntimeError(
                    f"Arguments 'inc_x' and 'inc_y' are needed for the specified domain_type '{domain_type}'")

        else:
            raise RuntimeError(f"Unsupported domain_type: '{domain_type}'")

        return None
