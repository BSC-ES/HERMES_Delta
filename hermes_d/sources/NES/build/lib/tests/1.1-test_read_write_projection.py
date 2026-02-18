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

result_path = "Times_test_1.1_read_write_projection_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'write'],
                      columns=['1.1.1.Regular', '1.1.2.Rotated', '1.1.3.Points', '1.1.4.Points_GHOST',
                               '1.1.5.LCC', '1.1.6.Mercator'])

# ======================================================================================================================
# =============================================     REGULAR     ========================================================
# ======================================================================================================================

test_name = '1.1.1.Regular'
if rank == 0:
    print(test_name)
comm.Barrier()

# Original path: /gpfs/scratch/bsc32/bsc32538/original_files/franco_interp.nc
# Regular lat-lon grid from MONARCH
path = '/gpfs/projects/bsc32/models/NES_tutorial_data/franco_interp.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, comm=comm, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
variables = ['sconcno2']
nessy.keep_vars(variables)
nessy.load()

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =============================================     ROTATED     ========================================================
# ======================================================================================================================

test_name = '1.1.2.Rotated'
if rank == 0:
    print(test_name)

# Rotated grid from MONARCH
path = '/gpfs/projects/bsc32/models/NES_tutorial_data/O3_all-000_2021080300.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, comm=comm, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
variables = ['O3_all']
nessy.keep_vars(variables)
nessy.load()

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


# ======================================================================================================================
# =============================================     LCC     ============================================================
# ======================================================================================================================

test_name = '1.1.5.LCC'
if rank == 0:
    print(test_name)

# Original path: /esarchive/exp/snes/a5g1/ip/daily_max/sconco3/sconco3_2022111500.nc
# LCC grid with a coverage over the Iberian Peninsula (4x4km)
path = '/gpfs/projects/bsc32/models/NES_tutorial_data/sconco3_2022111500.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, comm=comm, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
nessy.load()

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =============================================    MERCATOR     ========================================================
# ======================================================================================================================

test_name = '1.1.6.Mercator'
if rank == 0:
    print(test_name)

# Original path: None (generated with NES)
# Mercator grid
path = '/gpfs/projects/bsc32/models/NES_tutorial_data/mercator_grid.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, comm=comm, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
nessy.load()

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)

# ======================================================================================================================
# =============================================     POINTS     =========================================================
# ======================================================================================================================

test_name = '1.1.3.Points'
if rank == 0:
    print(test_name)

# Original path: /esarchive/obs/nilu/ebas/daily/pm10/pm10_201507.nc
# Points grid from EBAS network
path = '/gpfs/projects/bsc32/models/NES_tutorial_data/pm10_201507.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, comm=comm, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
nessy.load()

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =============================================     POINTS GHOST     ===================================================
# ======================================================================================================================

test_name = '1.1.4.Points_GHOST'
if rank == 0:
    print(test_name)

path = '/gpfs/projects/bsc32/AC_cache/obs/ghost/EBAS/1.4/hourly/sconco3/sconco3_201906.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, comm=comm, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
nessy.load()

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
