#!/usr/bin/env python
import os
import tempfile
import unittest

import numpy as np
from configargparse import ArgumentTypeError

from hermes_d.utilities import parse_vertical_description

PRECISION = np.float64


class TestParsing(unittest.TestCase):
    def test_parse_vertical_description_from_list(self):
        result = parse_vertical_description("48, 104, 168")

        np.testing.assert_array_equal(result, np.array([48, 104, 168], dtype=PRECISION))

    def test_parse_vertical_description_from_single_value(self):
        result = parse_vertical_description("20000")

        np.testing.assert_array_equal(result, np.array([20000], dtype=PRECISION))

    def test_parse_vertical_description_from_csv(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        try:
            tmp.write("Ilayer,height_magl\n1,39\n2,78\n3,119\n")
            tmp.close()

            result = parse_vertical_description(tmp.name)

            np.testing.assert_array_equal(result, np.array([39, 78, 119], dtype=PRECISION))
        finally:
            if not tmp.closed:
                tmp.close()
            os.remove(tmp.name)

    def test_parse_vertical_description_csv_requires_height_magl(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        try:
            tmp.write("Ilayer,height\n1,39\n")
            tmp.close()

            with self.assertRaises(ArgumentTypeError):
                parse_vertical_description(tmp.name)
        finally:
            if not tmp.closed:
                tmp.close()
            os.remove(tmp.name)

    def test_parse_vertical_description_csv_requires_numeric_height_magl(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
        try:
            tmp.write("Ilayer,height_magl\n1,not_a_height\n")
            tmp.close()

            with self.assertRaises(ArgumentTypeError):
                parse_vertical_description(tmp.name)
        finally:
            if not tmp.closed:
                tmp.close()
            os.remove(tmp.name)


if __name__ == "__main__":
    unittest.main()
