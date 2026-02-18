#!/usr/bin/env python

# Copyright 2025 Earth Sciences Department, BSC-CNS
#
# This file is part of HERMES.
#
# HERMES is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HERMES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HERMES. If not, see <http://www.gnu.org/licenses/>.

from setuptools import find_packages
from setuptools import setup


def read_version():
    with open("hermes_d/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name="HERMES_Delta",
    license="GNU GPL v3",
    # platforms=["GNU/Linux Debian"],
    version=read_version(),
    description="HERMES_Delta",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Carles Tena Medina",
    author_email="carles.tena@bsc.es",
    url="https://earth.bsc.es/gitlab/nextgeneu/HERMES",

    keywords=["emissions", "emission model", "top-down", "downscaling", "mocage", "cmaq", "monarch", "wrf-chem",
              "atmospheric composition", "air quality", "earth science", "SEIE", "Spain"],
    install_requires=[
        "numpy",
        "netCDF4>=1.3.1",
        "geopandas",
        "configargparse",
        "mpi4py",
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ],
    package_data={"": [
        "README.rst",
        "CHANGELOG.rst",
        "LICENSE.rst",
    ]
    },
    data_files=[(".", ["LICENSE.rst", "CHANGELOG.rst", ]),
                ("hermes_d/config", ["hermes_d/config/arguments_description.csv"])
                ],

    include_package_data=True,

    entry_points={
        "console_scripts": [
            "hermes_delta = hermes_d.hermes:run",
        ],
    },
)
