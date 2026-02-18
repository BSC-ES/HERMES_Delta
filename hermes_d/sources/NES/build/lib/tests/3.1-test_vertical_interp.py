#!/usr/bin/env python
import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'T'

result_path = "Times_test_3.1_vertical_interp_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate'],
                      columns=['3.1.1.Interp', '3.1.1.Exterp'])

# ======================================================================================================================
# ===============================================  VERTICAL INTERPOLATION  =============================================
# ======================================================================================================================

test_name = '3.1.1.Interp'
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()

# Original path: /esarchive/exp/monarch/a4dd/original_files/000/2022111512/MONARCH_d01_2022111512.nc
# Rotated grid from MONARCH
source_path = '/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc'

# Read source data
source_data = open_netcdf(path=source_path, info=True)

# Select time and load variables
source_data.keep_vars(['O3', 'mid_layer_height_agl'])
source_data.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# INTERPOLATE
st_time = timeit.default_timer()
source_data.vertical_var_name = 'mid_layer_height_agl'
level_list = [0., 50., 100., 250., 500., 750., 1000., 2000., 3000., 5000.]
interp_nes = source_data.interpolate_vertical(level_list, info=True, kind='linear', extrapolate=None)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===============================================  VERTICAL INTERPOLATION  =============================================
# ======================================================================================================================

test_name = '3.1.1.Exterp'
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()

# Original path: /esarchive/exp/monarch/a4dd/original_files/000/2022111512/MONARCH_d01_2022111512.nc
# Rotated grid from MONARCH
source_path = '/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc'

# Read source data
source_data = open_netcdf(path=source_path, info=True)

# Select time and load variables
source_data.keep_vars(['O3', 'mid_layer_height_agl'])
source_data.load()

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# INTERPOLATE
st_time = timeit.default_timer()
source_data.vertical_var_name = 'mid_layer_height_agl'
level_list = [0., 50., 100., 250., 500., 750., 1000., 2000., 3000., 5000., 21000, 25000, 30000, 40000, 50000]
interp_nes = source_data.interpolate_vertical(level_list, info=True, kind='linear', extrapolate=True)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
