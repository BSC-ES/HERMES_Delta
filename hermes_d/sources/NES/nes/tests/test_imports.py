import unittest


class TestImports(unittest.TestCase):
    """
    Unit tests to verify the availability and correct import of external Python packages required by the HERMES system.

    Each test ensures that a particular package or set of functionalities is importable in the current environment.
    Failures in these tests typically indicate that required dependencies are not properly installed or available.
    """

    def test_imports(self):
        imports_to_test = [
            "sys", "os", "time", "timeit", "math", "calendar", "datetime",
            "warnings", "geopandas", "pandas", "numpy", "shapely",
            "mpi4py", "netCDF4", "pyproj", "configargparse", "filelock",
            "pytz",  "eccodes", "scipy", "nes", "pytest", "dateutil",
            "rasterio", "hashlib", "sqlite3", ]

        for module_name in imports_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_eccodes(self):
        """
        Test that the `eccodes` library and its main GRIB manipulation functions can be successfully imported.

        This includes:
        - Creating a GRIB message from a sample file.
        - Accessing GRIB keys using a keys iterator.
        - Getting and setting values.
        - Writing and releasing GRIB messages.

        The test ensures that all critical `eccodes` functions are importable and accessible.
        """
        try:
            import eccodes
            from eccodes import codes_grib_new_from_file
            from eccodes import codes_keys_iterator_new
            from eccodes import codes_keys_iterator_next
            from eccodes import codes_keys_iterator_get_name
            from eccodes import codes_get_string
            from eccodes import codes_keys_iterator_delete
            from eccodes import codes_clone
            from eccodes import codes_set
            from eccodes import codes_set_values
            from eccodes import codes_write
            from eccodes import codes_release
            from eccodes import codes_samples_path
            import os
            os.path.join(codes_samples_path(), "GRIB2.tmpl")

            print("Eccodes: ", eccodes.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_geopandas(self):
        try:
            import geopandas
            from geopandas import sjoin_nearest
            from geopandas import GeoDataFrame
            print("GeoPandas: ", geopandas.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_pandas(self):
        try:
            import pandas
            print("Pandas: ", pandas.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_numpy(self):
        try:
            import numpy
            print("NumPy: ", numpy.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_shapely(self):
        try:
            import shapely
            print("Shapely: ", shapely.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_mpi(self):
        try:
            import mpi4py
            print("mpi4py: ", mpi4py.__version__)
            from mpi4py import MPI
            print("MPI Vendor: ", MPI.get_vendor())
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_netcdf4(self):
        try:
            import netCDF4
            print("netCDF4 version:", netCDF4.__version__)
            print("HDF5 version:", netCDF4.__hdf5libversion__)
            print("NetCDF library version:", netCDF4.__netcdf4libversion__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_netcdf4_parallel(self):
        import os
        try:
            from mpi4py import MPI
            import numpy as np
            from netCDF4 import Dataset

            nc = Dataset("parallel_test.nc", "w", parallel=True, comm=MPI.COMM_WORLD,  info=MPI.Info(),)
            nc.createDimension("x", 10)
            nc.close()
            self.assertTrue(os.path.exists("parallel_test.nc"))
        except (ImportError, RuntimeError, OSError) as e:
            self.fail(f"Parallel netCDF4 support not available: {e}")
        finally:
            if os.path.exists("parallel_test.nc"):
                os.remove("parallel_test.nc")

    def test_pyproj(self):
        try:
            import pyproj
            print("pyproj: ", pyproj.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_pytz(self):
        try:
            import pytz
            print("pytz: ", pytz.__version__)
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_rasterio(self):
        try:
            import rasterio
            from rasterio.windows import Window
            print("rasterio: ", rasterio.__version__)

        except ImportError as e:
            self.fail(f"Import error: {e}")


if __name__ == "__main__":
    unittest.main()
