#!/usr/bin/env python

import sys
from mpi4py import MPI
import pandas as pd
import timeit
from nes import create_nes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_1.2_create_projection_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['create', 'write'],
                      columns=['1.2.1.Regular', '1.2.2.Rotated', '1.2.3.LCC', '1.2.4.Mercator', '1.2.5.Global'])

# ======================================================================================================================
# =============================================     REGULAR     ========================================================
# ======================================================================================================================

test_name = '1.2.1.Regular'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.01
inc_lon = 0.01
n_lat = 100
n_lon = 100
nessy = create_nes(projection='regular', parallel_method=parallel_method,
                   lat_orig=lat_orig, lon_orig=lon_orig, inc_lat=inc_lat, inc_lon=inc_lon, n_lat=n_lat, n_lon=n_lon)

comm.Barrier()
result.loc['create', test_name] = timeit.default_timer() - st_time

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
# =============================================     ROTATED     ========================================================
# ======================================================================================================================

test_name = '1.2.2.Rotated'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
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

comm.Barrier()
result.loc['create', test_name] = timeit.default_timer() - st_time

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
# =============================================       LCC       ========================================================
# ======================================================================================================================

test_name = '1.2.3.LCC'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_1 = 37
lat_2 = 43
lon_0 = -3
lat_0 = 40
nx = 397
ny = 397
inc_x = 4000
inc_y = 4000
x_0 = -807847.688
y_0 = -797137.125
nessy = create_nes(projection='lcc', parallel_method=parallel_method,
                   lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                   nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0)

comm.Barrier()
result.loc['create', test_name] = timeit.default_timer() - st_time

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
# =============================================     MERCATOR    ========================================================
# ======================================================================================================================

test_name = '1.2.4.Mercator'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_ts = -1.5
lon_0 = -18.0
nx = 210
ny = 236
inc_x = 50000
inc_y = 50000
x_0 = -126017.5
y_0 = -5407460.0
nessy = create_nes(projection='mercator', parallel_method=parallel_method,
                   lat_ts=lat_ts, lon_0=lon_0, nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0)

comm.Barrier()
result.loc['create', test_name] = timeit.default_timer() - st_time

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
# ==============================================     GLOBAL     ========================================================
# ======================================================================================================================

test_name = '1.2.5.Global'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
inc_lat = 0.1
inc_lon = 0.1
nessy = create_nes(projection='global', parallel_method=parallel_method, inc_lat=inc_lat, inc_lon=inc_lon)

comm.Barrier()
result.loc['create', test_name] = timeit.default_timer() - st_time

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
