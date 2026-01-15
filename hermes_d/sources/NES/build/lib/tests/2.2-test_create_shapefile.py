#!/usr/bin/env python

import sys
import timeit
import pandas as pd
from mpi4py import MPI
import datetime
from nes import create_nes, open_netcdf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_2.2_create_shapefile_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate'],
                      columns=['2.2.1.Existing', '2.2.2.New_Regular',
                               '2.2.3.New_Rotated', '2.2.4.New_LCC', '2.2.5.New_Mercator'])

# ======================================================================================================================
# =====================================  CREATE SHAPEFILE FROM EXISTING GRID  ==========================================
# ======================================================================================================================

test_name = '2.2.1.Existing'
if rank == 0:
    print(test_name)

# Original path: /gpfs/scratch/bsc32/bsc32538/original_files/franco_interp.nc
# Regular lat-lon grid from MONARCH
path = '/gpfs/projects/bsc32/models/NES_tutorial_data/franco_interp.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(path=path, info=True, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# LOAD VARIABLES
nessy.load()

# CREATE SHAPEFILE
st_time = timeit.default_timer()
nessy.to_shapefile(path='regular_shp',
                   time=datetime.datetime(2019, 1, 1, 10, 0),
                   lev=0, var_list=['sconcno2'])
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM EXISTING GRID - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =====================================  CREATE SHAPEFILE FROM NEW REGULAR GRID  =======================================
# ======================================================================================================================

test_name = '2.2.2.New_Regular'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.1
inc_lon = 0.1
n_lat = 50
n_lon = 100
nessy = create_nes(comm=None, info=False, projection='regular',
                   lat_orig=lat_orig, lon_orig=lon_orig, inc_lat=inc_lat, inc_lon=inc_lon,
                   n_lat=n_lat, n_lon=n_lon)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE SHAPEFILE
st_time = timeit.default_timer()
nessy.create_shapefile()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM NEW REGULAR GRID - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =====================================  CREATE SHAPEFILE FROM NEW ROTATED GRID  =======================================
# ======================================================================================================================

test_name = '2.2.3.New_Rotated'
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
nessy = create_nes(comm=None, info=False, projection='rotated',
                   centre_lat=centre_lat, centre_lon=centre_lon,
                   west_boundary=west_boundary, south_boundary=south_boundary,
                   inc_rlat=inc_rlat, inc_rlon=inc_rlon)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE SHAPEFILE
st_time = timeit.default_timer()
nessy.create_shapefile()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM NEW ROTATED GRID - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =====================================  CREATE SHAPEFILE FROM NEW LCC GRID  ===========================================
# ======================================================================================================================

test_name = '2.2.4.New_LCC'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_1 = 37
lat_2 = 43
lon_0 = -3
lat_0 = 40
nx = 100
ny = 200
inc_x = 4000
inc_y = 4000
x_0 = -807847.688
y_0 = -797137.125
nessy = create_nes(comm=None, info=False, projection='lcc',
                   lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                   nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE SHAPEFILE
st_time = timeit.default_timer()
nessy.create_shapefile()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM NEW LCC GRID - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =====================================  CREATE SHAPEFILE FROM NEW MERCATOR GRID  ======================================
# ======================================================================================================================

test_name = '2-2.5.New_Mercator'
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_ts = -1.5
lon_0 = -18.0
nx = 100
ny = 50
inc_x = 50000
inc_y = 50000
x_0 = -126017.5
y_0 = -5407460.0
nessy = create_nes(comm=None, info=False, projection='mercator',
                   lat_ts=lat_ts, lon_0=lon_0, nx=nx, ny=ny,
                   inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CREATE SHAPEFILE
st_time = timeit.default_timer()
nessy.create_shapefile()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM NEW MERCATOR GRID - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
