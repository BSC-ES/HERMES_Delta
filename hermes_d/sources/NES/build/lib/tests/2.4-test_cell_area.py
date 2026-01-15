#!/usr/bin/env python

import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import create_nes, open_netcdf, calculate_geometry_area

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_2.4_cell_area_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['2.4.1.New_file_grid_area', '2.4.2.New_file_geometry_area',
                               '2.4.3.Existing_file_grid_area', '2.4.4.Existing_file_geometry_area'])

# ======================================================================================================================
# =====================================  CALCULATE CELLS AREA FROM NEW GRID  ===========================================
# ======================================================================================================================

test_name = "2.4.1.New_file_grid_area"
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_1 = 37
lat_2 = 43
lon_0 = -3
lat_0 = 40
nx = 20
ny = 40
inc_x = 4000
inc_y = 4000
x_0 = -807847.688
y_0 = -797137.125
nessy = create_nes(comm=None, info=False, projection='lcc',
                   lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                   nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CALCULATE AREA OF EACH CELL IN GRID
st_time = timeit.default_timer()
nessy.calculate_grid_area()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE GRID AREA
print('Rank {0:03d}: Calculate grid cell area: {1}'.format(rank, nessy.cell_measures['cell_area']))

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
# nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))

# EXPLORE GRID AREA
print('Rank {0:03d}: Write grid cell area: {1}'.format(rank, nessy.cell_measures['cell_area']))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

del nessy

# ======================================================================================================================
# =====================================  CALCULATE CELLS AREA FROM GEOMETRIES  =========================================
# ======================================================================================================================

test_name = "2.4.2.New_file_geometry_area"
if rank == 0:
    print(test_name)

# CREATE GRID
st_time = timeit.default_timer()
lat_1 = 37
lat_2 = 43
lon_0 = -3
lat_0 = 40
nx = 20
ny = 40
inc_x = 4000
inc_y = 4000
x_0 = -807847.688
y_0 = -797137.125
nessy = create_nes(comm=None, info=False, projection='lcc',
                   lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                   nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CALCULATE AREA OF EACH CELL POLYGON
st_time = timeit.default_timer()
nessy.create_shapefile()
geometry_list = nessy.shapefile['geometry'].values
geometry_area = calculate_geometry_area(geometry_list)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE GEOMETRIES AREA
print('Rank {0:03d}: Calculate geometry cell area: {1}'.format(rank, geometry_area))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# =====================================  CALCULATE CELLS AREA FROM EXISTING GRID  ======================================
# ======================================================================================================================

test_name = '2.4.3.Existing_file_grid_area'
if rank == 0:
    print(test_name)

# Original path: /esarchive/exp/monarch/a4dd/original_files/000/2022111512/MONARCH_d01_2022111512.nc
# Rotated grid from MONARCH
original_path = '/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(original_path, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CALCULATE AREA OF EACH CELL IN GRID
st_time = timeit.default_timer()
nessy.calculate_grid_area()
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE GRID AREA
print('Rank {0:03d}: Calculate grid cell area: {1}'.format(rank, nessy.cell_measures['cell_area']))

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
# nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()
del nessy

# ======================================================================================================================
# =====================================  CALCULATE CELLS AREA FROM GEOMETRIES FROM EXISTING GRID  ======================
# ======================================================================================================================

test_name = '2.4.4.Existing_file_geometry_area'
if rank == 0:
    print(test_name)

# Original path: /esarchive/exp/monarch/a4dd/original_files/000/2022111512/MONARCH_d01_2022111512.nc
# Rotated grid from MONARCH
original_path = '/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc'

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(original_path, parallel_method=parallel_method)
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CALCULATE AREA OF EACH CELL POLYGON
st_time = timeit.default_timer()
nessy.create_shapefile()
geometry_list = nessy.shapefile['geometry'].values
geometry_area = calculate_geometry_area(geometry_list)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# EXPLORE GEOMETRIES AREA
print('Rank {0:03d}: Calculate geometry cell area: {1}'.format(rank, geometry_area))

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()
del nessy

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
