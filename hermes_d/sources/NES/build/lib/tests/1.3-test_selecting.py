#!/usr/bin/env python
import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'
serial_write = True

result_path = "Times_test_1.3.Selecting_{0}_{1:03d}.csv".format(parallel_method, size)

result = pd.DataFrame(index=['read', 'calcul', 'write'],
                      columns=['1.3.1.LatLon', '1.3.2.Level', '1.3.3.Time', '1.3.4.Time_min', '1.3.5.Time_max'])

# NAMEE
src_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc"
var_list = ['O3']

# ======================================================================================================================
# ======================================      '1.3.1.LatLon'       =====================================================
# ======================================================================================================================
test_name = '1.3.1.Selecting_LatLon'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method, balanced=True)
nessy.keep_vars(var_list)
nessy.sel(lat_min=35, lat_max=45, lon_min=-9, lon_max=5)

nessy.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================      1.3.2.Level       =====================================================
# ======================================================================================================================
test_name = '1.3.2.Selecting_Level'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method)
nessy.keep_vars(var_list)
nessy.sel(lev_min=3, lev_max=5)

nessy.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================          1.3.3.Time       =====================================================
# ======================================================================================================================
test_name = '1.3.3.Selecting_Time'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method)
nessy.keep_vars(var_list)
nessy.sel(time_min=datetime(year=2022, month=11, day=16, hour=0),
          time_max=datetime(year=2022, month=11, day=16, hour=0))

nessy.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================      '1.3.4.Time_min'     =====================================================
# ======================================================================================================================
test_name = '1.3.4.Selecting_Time_min'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method)
nessy.keep_vars(var_list)
nessy.sel(time_min=datetime(year=2022, month=11, day=16, hour=0))

nessy.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================      '1.3.5.Time_max'     =====================================================
# ======================================================================================================================
test_name = '1.3.5.Selecting_Time_max'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method)
nessy.keep_vars(var_list)
nessy.sel(time_max=datetime(year=2022, month=11, day=16, hour=0))

nessy.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
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
