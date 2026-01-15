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

result_path = "Times_test_2.3_bounds_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['2.3.1.With_bounds', '2.3.2.Without_bounds', "2.3.3.Create_new",
                               "2.3.4.latlon_sel_create_bnds", "2.3.5.rotated_sel_create_bnds"])

# ======================================================================================================================
# =====================================  FILE WITH EXISTING BOUNDS  ====================================================
# ======================================================================================================================

test_name = "2.3.1.With_bounds"
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()
# Original path: /esarchive/exp/snes/a5s1/regional/3hourly/od550du/od550du-000_2021070612.nc
# Rotated grid for dust regional
path_1 = '/gpfs/projects/bsc32/models/NES_tutorial_data/od550du-000_2021070612.nc'
nessy_1 = open_netcdf(path=path_1, parallel_method=parallel_method, info=True)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# EXPLORE BOUNDS
st_time = timeit.default_timer()
print('FILE WITH EXISTING BOUNDS - Rank', rank, '-', 'Lat bounds', nessy_1.lat_bnds)
print('FILE WITH EXISTING BOUNDS - Rank', rank, '-', 'Lon bounds', nessy_1.lon_bnds)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
nessy_1.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy_2 = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)

# LOAD DATA AND EXPLORE BOUNDS
print('FILE WITH EXISTING BOUNDS AFTER WRITE - Rank', rank, '-', 'Lat bounds', nessy_2.lat_bnds)
print('FILE WITH EXISTING BOUNDS AFTER WRITE - Rank', rank, '-', 'Lon bounds', nessy_2.lon_bnds)

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===================================  FILE WITHOUT EXISTING BOUNDS  ===================================================
# ======================================================================================================================

test_name = '2.3.2.Without_bounds'
if rank == 0:
    print(test_name)

# Original path: /gpfs/scratch/bsc32/bsc32538/mr_multiplyby/OUT/stats_bnds/monarch/a45g/regional/daily_max/O3_all
# /O3_all-000_2021080300.nc Rotated grid from MONARCH
st_time = timeit.default_timer()
path_3 = "/gpfs/projects/bsc32/models/NES_tutorial_data/O3_all-000_2021080300.nc"
nessy_3 = open_netcdf(path=path_3, parallel_method=parallel_method, info=True)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE BOUNDS
st_time = timeit.default_timer()
nessy_3.create_spatial_bounds()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE BOUNDS
print('FILE WITHOUT EXISTING BOUNDS - Rank', rank, '-', 'Lat bounds', nessy_3.lat_bnds)
print('FILE WITHOUT EXISTING BOUNDS - Rank', rank, '-', 'Lon bounds', nessy_3.lon_bnds)

# WRITE
st_time = timeit.default_timer()
nessy_3.to_netcdf('/tmp/bounds_file_2.nc', info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy_4 = open_netcdf('/tmp/bounds_file_2.nc', info=True)

# LOAD DATA AND EXPLORE BOUNDS
print('FILE WITH EXISTING BOUNDS AFTER WRITE - Rank', rank, '-', 'Lat bounds', nessy_4.lat_bnds)
print('FILE WITH EXISTING BOUNDS AFTER WRITE - Rank', rank, '-', 'Lon bounds', nessy_4.lon_bnds)

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ====================================  CREATE NES REGULAR LAT-LON  ====================================================
# ======================================================================================================================

test_name = "2.3.3.Create_new"
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.2
inc_lon = 0.2
n_lat = 100
n_lon = 100
nessy_5 = create_nes(comm=None, parallel_method=parallel_method, info=True, projection='regular',
                     lat_orig=lat_orig, lon_orig=lon_orig,
                     inc_lat=inc_lat, inc_lon=inc_lon,
                     n_lat=n_lat, n_lon=n_lon)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE BOUNDS
st_time = timeit.default_timer()
nessy_5.create_spatial_bounds()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE BOUNDS
print('FROM NEW GRID - Rank', rank, '-', 'Lat bounds', nessy_5.lat_bnds)
print('FROM NEW GRID - Rank', rank, '-', 'Lon bounds', nessy_5.lon_bnds)

# WRITE
st_time = timeit.default_timer()
nessy_5.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy_6 = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)

# LOAD DATA AND EXPLORE BOUNDS
print('FROM NEW GRID AFTER WRITE - Rank', rank, '-', 'Lat bounds', nessy_6.lat_bnds)
print('FROM NEW GRID AFTER WRITE - Rank', rank, '-', 'Lon bounds', nessy_6.lon_bnds)

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


# ======================================================================================================================
# ================================  REGULAR LAT-LON SEL THEN CREATE BOUNDS =============================================
# ======================================================================================================================

test_name = "2.3.4.latlon_sel_create_bnds"
if rank == 0:
    print(test_name)

# USE SAME GRID SETTING AS 2.3.3
nessy_7 = create_nes(comm=None, parallel_method=parallel_method, info=True, projection='regular',
                     lat_orig=lat_orig, lon_orig=lon_orig,
                     inc_lat=inc_lat, inc_lon=inc_lon,
                     n_lat=n_lat, n_lon=n_lon)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# SEL
nessy_7.sel(lat_min=50, lat_max=60, lon_min=10, lon_max=20)

# CREATE BOUNDS
st_time = timeit.default_timer()
nessy_7.create_spatial_bounds()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE BOUNDS
print('FROM NEW GRID - Rank', rank, '-', 'Lat bounds', nessy_7.lat_bnds)
print('FROM NEW GRID - Rank', rank, '-', 'Lon bounds', nessy_7.lon_bnds)

# Check lon_bnds
if nessy_7.lon_bnds['data'].shape != (52, 2):
    raise Exception("Wrong lon_bnds.")

# WRITE
st_time = timeit.default_timer()
nessy_7.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy_8 = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), info=True)

# LOAD DATA AND EXPLORE BOUNDS
print('FROM NEW GRID AFTER WRITE - Rank', rank, '-', 'Lat bounds', nessy_8.lat_bnds)
print('FROM NEW GRID AFTER WRITE - Rank', rank, '-', 'Lon bounds', nessy_8.lon_bnds)

# Check lon_bnds
if nessy_8.lon_bnds['data'].shape != (52, 2):
    raise Exception("Wrong lon_bnds.")

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


# ======================================================================================================================
# ================================  ROTATED SEL THEN CREATE BOUNDS =============================================
# ======================================================================================================================

test_name = "2.3.5.rotated_sel_create_bnds"
if rank == 0:
    print(test_name)

# USE FILE AS 2.3.2

# Original path: /gpfs/scratch/bsc32/bsc32538/mr_multiplyby/OUT/stats_bnds/monarch/a45g/regional/daily_max/O3_all
# /O3_all-000_2021080300.nc Rotated grid from MONARCH
st_time = timeit.default_timer()
path_9 = "/gpfs/projects/bsc32/models/NES_tutorial_data/O3_all-000_2021080300.nc"
nessy_9 = open_netcdf(path=path_9, parallel_method=parallel_method, info=True)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# SEL
nessy_9.sel(lat_min=50, lat_max=60, lon_min=10, lon_max=15)

# CREATE BOUNDS
st_time = timeit.default_timer()
nessy_9.create_spatial_bounds()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE BOUNDS
print('FILE WITHOUT EXISTING BOUNDS - Rank', rank, '-', 'Lat bounds', nessy_9.lat_bnds)
print('FILE WITHOUT EXISTING BOUNDS - Rank', rank, '-', 'Lon bounds', nessy_9.lon_bnds)

# Check lon_bnds
if nessy_9.lon_bnds['data'].shape[0:2] != nessy_9.lon['data'].shape:
    raise Exception("Wrong lon_bnds.")

# WRITE
st_time = timeit.default_timer()
nessy_9.to_netcdf('/tmp/bounds_file_9.nc', info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy_10 = open_netcdf('/tmp/bounds_file_9.nc', info=True)

# LOAD DATA AND EXPLORE BOUNDS
print('FILE WITH EXISTING BOUNDS AFTER WRITE - Rank', rank, '-', 'Lat bounds', nessy_10.lat_bnds)
print('FILE WITH EXISTING BOUNDS AFTER WRITE - Rank', rank, '-', 'Lon bounds', nessy_10.lon_bnds)

# Check lon_bnds
if nessy_10.lon_bnds['data'].shape[0:2] != nessy_10.lon['data'].shape:
    raise Exception("Wrong lon_bnds.")

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
