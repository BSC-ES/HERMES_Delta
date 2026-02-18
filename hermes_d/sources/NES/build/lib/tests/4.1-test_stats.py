#!/usr/bin/env python

import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_4.1_daily_stats_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['4.1.1.Mean'])

# ======================================================================================================================
# ==============================================  CALCULATE DAILY MEAN  ================================================
# ======================================================================================================================

test_name = '4.1.1.Mean'
if rank == 0:
    print(test_name)

# Original path: /esarchive/exp/monarch/a4dd/original_files/000/2022111512/MONARCH_d01_2022111512.nc
# Rotated grid from MONARCH
cams_file = '/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=cams_file, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
nessy.keep_vars('O3')
nessy.load()

# CALCULATE MEAN
st_time = timeit.default_timer()
nessy.daily_statistic(op="mean")
print(nessy.variables['O3']['cell_methods'])
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ==========================================  CALCULATE 8-HOUR ROLLING MEAN  ===========================================
# ======================================================================================================================

test_name = '4.1.2.Rolling_Mean'
if rank == 0:
    print(test_name)

# Original path: /esarchive/exp/monarch/a4dd/original_files/000/2022111512/MONARCH_d01_2022111512.nc
# Rotated grid from MONARCH
cams_file = '/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=cams_file, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CALCULATE MEAN
st_time = timeit.default_timer()
rolling_mean = nessy.rolling_mean(var_list='O3', hours=8)
print(rolling_mean.variables['O3']['data'])
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
