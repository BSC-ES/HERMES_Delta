#!/usr/bin/env python

import sys
from mpi4py import MPI
import pandas as pd
import timeit
from nes import open_netcdf, from_shapefile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'
serial_write = False

result_path = "Times_test_2.1_spatial_join_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['2.1.1.Existing_file_centroid', '2.1.2.New_file_centroid',
                               '2.1.3.Existing_file_nearest', '2.1.4.New_file_nearest',
                               '2.1.5.Existing_file_intersection', '2.1.6.New_file_intersection'])

# ===== PATH TO MASK ===== #
# Timezones
# shapefile_path = '/gpfs/projects/bsc32/models/NES_tutorial_data/timezones_2021c/timezones_2021c.shp'
# shapefile_var_list = ['tzid']
# str_len = 32
# Country ISO codes
shapefile_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/gadm_country_mask/gadm_country_ISO3166.shp"
shapefile_var_list = ['ISO']
str_len = 3

# Original path: /gpfs/scratch/bsc32/bsc32538/original_files/franco_interp.nc
# Regular lat-lon grid from MONARCH
original_path = '/gpfs/projects/bsc32/models/NES_tutorial_data/franco_interp.nc'
# CAMS_Global
# original_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/nox_no_201505.nc"

# ======================================================================================================================
# ===================================     CENTROID EXISTING FILE     ===================================================
# ======================================================================================================================

test_name = '2.1.1.Existing_file_centroid'
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(original_path, parallel_method=parallel_method)
nessy.variables = {}
nessy.create_shapefile()
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# SPATIAL JOIN
# Method can be centroid, nearest and intersection
st_time = timeit.default_timer()
nessy.spatial_join(shapefile_path, method='centroid', var_list=shapefile_var_list, info=True)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()

# ADD Var
for var_name in shapefile_var_list:
    data = nessy.shapefile[var_name].values.reshape([nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]])
    nessy.variables[var_name] = {'data': data, 'dtype': str}
nessy.set_strlen(str_len)
comm.Barrier()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), serial=serial_write)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), parallel_method=parallel_method)
nessy.load()

# REWRITE
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}_2.nc".format(size), serial=serial_write)

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}_2.nc".format(size), parallel_method=parallel_method)
nessy.load()

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===================================     CENTROID FROM NEW FILE     ===================================================
# ======================================================================================================================

test_name = '2.1.2.New_file_centroid'
if rank == 0:
    print(test_name)

# DEFINE PROJECTION
st_time = timeit.default_timer()
projection = 'regular'
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.2
inc_lon = 0.2
n_lat = 100
n_lon = 100

# SPATIAL JOIN
# Method can be centroid, nearest and intersection
nessy = from_shapefile(shapefile_path, method='centroid', projection=projection,
                       lat_orig=lat_orig, lon_orig=lon_orig,
                       inc_lat=inc_lat, inc_lon=inc_lon,
                       n_lat=n_lat, n_lon=n_lon)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM SHAPEFILE - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

# ADD Var
for var_name in shapefile_var_list:
    data = nessy.shapefile[var_name].values.reshape([nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]])
    nessy.variables[var_name] = {'data': data, 'dtype': str}
nessy.set_strlen(str_len)

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), serial=serial_write, info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), parallel_method=parallel_method)
nessy.load()

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===================================     NEAREST EXISTING FILE      ===================================================
# ======================================================================================================================

test_name = '2.1.3.Existing_file_nearest'
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(original_path, parallel_method=parallel_method)
nessy.variables = {}
nessy.create_shapefile()
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# SPATIAL JOIN
# Method can be centroid, nearest and intersection
st_time = timeit.default_timer()
nessy.spatial_join(shapefile_path, method='nearest', var_list=shapefile_var_list, info=True)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('SPATIAL JOIN - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

# ADD Var
for var_name in shapefile_var_list:
    data = nessy.shapefile[var_name].values.reshape([nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]])
    nessy.variables[var_name] = {'data': data, 'dtype': str}
nessy.set_strlen(str_len)

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), serial=serial_write, info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), parallel_method=parallel_method)
nessy.load()

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===================================     NEAREST FROM NEW FILE      ===================================================
# ======================================================================================================================

test_name = '2.1.4.New_file_nearest'
if rank == 0:
    print(test_name)

# DEFINE PROJECTION
st_time = timeit.default_timer()
projection = 'regular'
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.2
inc_lon = 0.2
n_lat = 100
n_lon = 100

# SPATIAL JOIN
# Method can be centroid, nearest and intersection
nessy = from_shapefile(shapefile_path, method='nearest', projection=projection,
                       lat_orig=lat_orig, lon_orig=lon_orig,
                       inc_lat=inc_lat, inc_lon=inc_lon,
                       n_lat=n_lat, n_lon=n_lon)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM SHAPEFILE - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

# ADD Var
for var_name in shapefile_var_list:
    data = nessy.shapefile[var_name].values.reshape([nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]])
    nessy.variables[var_name] = {'data': data, 'dtype': str}
nessy.set_strlen(str_len)

# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), serial=serial_write, info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), parallel_method=parallel_method)
nessy.load()

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()


# ======================================================================================================================
# ===================================   INTERSECTION EXISTING FILE   ===================================================
# ======================================================================================================================

test_name = '2.1.5.Existing_file_intersection'
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()
nessy = open_netcdf(original_path, parallel_method=parallel_method)
nessy.variables = {}
nessy.create_shapefile()
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# SPATIAL JOIN
# Method can be centroid, nearest and intersection
st_time = timeit.default_timer()
nessy.spatial_join(shapefile_path, method='intersection', var_list=shapefile_var_list, info=True)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('SPATIAL JOIN - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

# ADD Var
for var_name in shapefile_var_list:
    data = nessy.shapefile[var_name].values.reshape([nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]])
    nessy.variables[var_name] = {'data': data, 'dtype': str}
nessy.set_strlen(str_len)

# WRITE
st_time = timeit.default_timer()
nessy.set_strlen(strlen=str_len)
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), serial=serial_write, info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), parallel_method=parallel_method)
nessy.load()

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ===================================   INTERSECTION FROM NEW FILE   ===================================================
# ======================================================================================================================

test_name = '2.1.6.New_file_intersection'
if rank == 0:
    print(test_name)

# DEFINE PROJECTION
st_time = timeit.default_timer()
projection = 'regular'
lat_orig = 41.1
lon_orig = 1.8
inc_lat = 0.2
inc_lon = 0.2
n_lat = 100
n_lon = 100

# SPATIAL JOIN
# Method can be centroid, nearest and intersection
nessy = from_shapefile(shapefile_path, method='intersection', projection=projection,
                       lat_orig=lat_orig, lon_orig=lon_orig,
                       inc_lat=inc_lat, inc_lon=inc_lon,
                       n_lat=n_lat, n_lon=n_lon)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time
print('FROM SHAPEFILE - Rank {0:03d} - Shapefile: \n{1}'.format(rank, nessy.shapefile))

# ADD Var
for var_name in shapefile_var_list:
    data = nessy.shapefile[var_name].values.reshape([nessy.lat['data'].shape[0], nessy.lon['data'].shape[-1]])
    nessy.variables[var_name] = {'data': data, 'dtype': str}
nessy.set_strlen(str_len)
# WRITE
st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), serial=serial_write, info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

# REOPEN
nessy = open_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size), parallel_method=parallel_method)
nessy.load()

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
