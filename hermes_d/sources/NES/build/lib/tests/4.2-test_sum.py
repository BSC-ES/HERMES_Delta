#!/usr/bin/env python

import sys
from mpi4py import MPI
import pandas as pd
import timeit
import numpy as np
from nes import create_nes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_4.2_sum_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['4.2.1.Sum'])

# ======================================================================================================================
# ===================================     CENTROID FROM NEW FILE     ===================================================
# ======================================================================================================================

test_name = '4.2.1.Sum'

if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
projection = 'regular'
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.2
inc_lon = 0.2
n_lat = 100
n_lon = 100
nessy = create_nes(projection=projection, lat_orig=lat_orig, lon_orig=lon_orig, inc_lat=inc_lat, inc_lon=inc_lon,
                   n_lat=n_lat, n_lon=n_lon)

# ADD VARIABLES
nessy.variables = {'var_aux': {'data': np.ones((len(nessy.time), len(nessy.lev['data']),
                                                nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]))}}

# CREATE GRID WITH COPY
nessy_2 = nessy.copy(copy_vars=True)

# ADD VARIABLES
for var_name in nessy_2.variables.keys():
    nessy_2.variables[var_name]['data'] *= 2

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# SUM
st_time = timeit.default_timer()
nessy_3 = nessy + nessy_2
print('Sum result', nessy_3.variables['var_aux']['data'])

comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
nessy_3.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
