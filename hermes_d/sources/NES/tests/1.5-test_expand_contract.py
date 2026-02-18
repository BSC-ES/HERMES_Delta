#!/usr/bin/env python
import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf, create_nes


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'
serial_write = True

result_path = "Times_test_1.5_Expand_Contract_{0}_{1:03d}.csv".format(parallel_method, size)

result = pd.DataFrame(index=['read', 'calcul', 'write'],
                      columns=['1.5.1.Expand_NoLoaded', '1.5.2.Expand_Loaded', '1.5.3.Expand_Contract',
                               '1.5.4.Expand_Contract_Rotated'])

# Global GFAS
# src_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/ga_20220101.nc"
src_path = "/Users/ctena/PycharmProjects/NES/tests/ga_20220101.nc"
var_list = ['so2', "pm25", 'nox']

# ======================================================================================================================
# ======================================      '1.5.1.Expand_NoLoaded'       ============================================
# ======================================================================================================================
test_name = '1.5.1.Expand_NoLoaded'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method, balanced=True)
print(f"1 Read {nessy.read_axis_limits}")
print(f"1 Write {nessy.write_axis_limits}")

nessy.keep_vars(var_list)
nessy.sel(lat_min=35, lat_max=45, lon_min=-9, lon_max=5)
print(f"2 Read {nessy.read_axis_limits}")
print(f"2 Write {nessy.write_axis_limits}")

nessy.expand(n_cells=10)
print(f"3 Read {nessy.read_axis_limits}")
print(f"3 Write {nessy.write_axis_limits}")

nessy.load()

# print(f"Rank {nessy.rank} -> Lon: {nessy.lon}")
# print(f"Rank {nessy.rank} -> Lon BNDS: {nessy.lon_bnds}")
# print(f"Rank {nessy.rank} -> FULL Lon: {nessy.get_full_longitudes()}")

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
print(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================      '1.5.2.Expand_Loaded'       ==============================================
# ======================================================================================================================
test_name = '1.5.2.Expand_Loaded'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method, balanced=True)
nessy.keep_vars(var_list)
nessy.sel(lat_min=35, lat_max=45, lon_min=-9, lon_max=5)

nessy.load()

nessy.expand(n_cells=10)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
print(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


# ======================================================================================================================
# ======================================      '1.5.3.Expand_Contract'       ============================================
# ======================================================================================================================
test_name = '1.5.3.Expand_Contract'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method, balanced=True)

nessy.keep_vars(var_list)
nessy.sel(lat_min=35, lat_max=45, lon_min=-9, lon_max=5)

nessy.expand(n_cells=10)

nessy.load()

nessy.contract(n_cells=10)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
print(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================      '1.5.4.Expand_Contract_Rotated'       ====================================
# ======================================================================================================================
test_name = '1.5.4.Expand_Contract_Rotated'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
centre_lat = 51
centre_lon = 10
west_boundary = -35
south_boundary = -27
inc_rlat = 0.2
inc_rlon = 0.2
nessy = create_nes(projection='rotated', parallel_method=parallel_method,
                   centre_lat=centre_lat, centre_lon=centre_lon,
                   west_boundary=west_boundary, south_boundary=south_boundary,
                   inc_rlat=inc_rlat, inc_rlon=inc_rlon)

nessy.create_spatial_bounds()

nessy.expand(n_cells=10)

nessy.contract(n_cells=10)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
print(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
