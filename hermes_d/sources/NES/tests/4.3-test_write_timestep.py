#!/usr/bin/env python

import sys
from mpi4py import MPI
import pandas as pd
import timeit
from datetime import datetime, timedelta
import numpy as np
from nes import create_nes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_4.3_write_time_step_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['4.3.1.Parallel_Write', '4.3.2.Serial_Write'])

# ======================================================================================================================
# ===================================         PARALLEL WRITE         ===================================================
# ======================================================================================================================

test_name = '4.3.1.Parallel_Write'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()
# CREATE GRID
centre_lat = 51
centre_lon = 10
west_boundary = -35
south_boundary = -27
inc_rlat = 0.2
inc_rlon = 0.2
nessy = create_nes(comm=None, info=False, projection='rotated',
                   centre_lat=centre_lat, centre_lon=centre_lon,
                   west_boundary=west_boundary, south_boundary=south_boundary,
                   inc_rlat=inc_rlat, inc_rlon=inc_rlon)

# ADD VARIABLES
nessy.variables = {'var1': {'data': None, 'units': 'kg.s-1', 'dtype': np.float32},
                   'var2': {'data': None, 'units': 'kg.s-1', 'dtype': np.float32}}
time_list = [datetime(year=2023, month=1, day=1) + timedelta(hours=x) for x in range(24)]
nessy.set_time(time_list)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name + '.nc', keep_open=True, info=False)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# CALCULATE & APPEND
result.loc['calculate', test_name] = 0

for i_time, time_aux in enumerate(time_list):
    # CALCULATE
    st_time = timeit.default_timer()

    nessy.variables['var1']['data'] = np.ones((1, 1, nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1])) * i_time
    nessy.variables['var2']['data'] = np.ones((1, 1, nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1])) * i_time

    comm.Barrier()
    result.loc['calculate', test_name] += timeit.default_timer() - st_time

    # APPEND
    st_time = timeit.default_timer()
    nessy.append_time_step_data(i_time)
    comm.Barrier()
    if i_time == len(time_list) - 1:
        nessy.close()
    result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===================================         SERIAL   WRITE         ===================================================
# ======================================================================================================================

test_name = '4.3.2.Serial_Write'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()
# CREATE GRID
centre_lat = 51
centre_lon = 10
west_boundary = -35
south_boundary = -27
inc_rlat = 0.2
inc_rlon = 0.2
nessy = create_nes(comm=None, info=False, projection='rotated',
                   centre_lat=centre_lat, centre_lon=centre_lon,
                   west_boundary=west_boundary, south_boundary=south_boundary,
                   inc_rlat=inc_rlat, inc_rlon=inc_rlon)

# ADD VARIABLES
nessy.variables = {'var1': {'data': None, 'units': 'kg.s-1', 'dtype': np.float32},
                   'var2': {'data': None, 'units': 'kg.s-1', 'dtype': np.float32}}
time_list = [datetime(year=2023, month=1, day=1) + timedelta(hours=x) for x in range(24)]
nessy.set_time(time_list)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name + '.nc', keep_open=True, info=False, serial=True)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# CALCULATE & APPEND
result.loc['calculate', test_name] = 0

for i_time, time_aux in enumerate(time_list):
    # CALCULATEATE
    st_time = timeit.default_timer()

    nessy.variables['var1']['data'] = np.ones((1, 1, nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1])) * i_time
    nessy.variables['var2']['data'] = np.ones((1, 1, nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1])) * i_time

    comm.Barrier()
    result.loc['calculate', test_name] += timeit.default_timer() - st_time

    # APPEND
    st_time = timeit.default_timer()
    nessy.append_time_step_data(i_time)
    comm.Barrier()
    if i_time == len(time_list) - 1:
        nessy.close()
    result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
