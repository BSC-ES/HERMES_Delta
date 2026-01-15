#!/usr/bin/env python
import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf, create_nes
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'T'

result_path = "Times_test_3.2_horiz_interp_bilinear_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['3.2.1.Only interp', '3.2.2.Create_WM', "3.2.3.Use_WM", "3.2.4.Read_WM"])

# NAMEE
src_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc"
var_list = ['O3']

# ======================================================================================================================
# ======================================        Only interp        =====================================================
# ======================================================================================================================
test_name = '3.2.1.NN_Only interp'
if rank == 0:
    print(test_name)
    sys.stdout.flush()

# READING
st_time = timeit.default_timer()

# Source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Destination Grid
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
dst_nes = create_nes(comm=None, info=False, projection='lcc', lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                     nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0, parallel_method=parallel_method,
                     times=src_nes.get_full_times())
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()

interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='NN')
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================         Create_WM        =====================================================
# ======================================================================================================================
test_name = '3.2.2.NN_Create_WM'
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()

# Read source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Destination Grid
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
dst_nes = create_nes(comm=None, info=False, projection='lcc', lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                     nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0, parallel_method=parallel_method,
                     times=src_nes.get_full_times())
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# Cleaning WM
if os.path.exists("NN_WM_NAMEE_to_IP.nc") and rank == 0:
    os.remove("NN_WM_NAMEE_to_IP.nc")
comm.Barrier()

st_time = timeit.default_timer()

wm_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='NN', info=True,
                                        weight_matrix_path="NN_WM_NAMEE_to_IP.nc", only_create_wm=True)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================           Use_WM         =====================================================
# ======================================================================================================================
test_name = "3.2.3.NN_Use_WM"
if rank == 0:
    print(test_name)

# READING
st_time = timeit.default_timer()

# Source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Destination Grid
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

dst_nes = create_nes(comm=None, info=False, projection='lcc', lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                     nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0, parallel_method=parallel_method,
                     times=src_nes.get_full_times())
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()

interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='NN', wm=wm_nes)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "_{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================           Read_WM         =====================================================
# ======================================================================================================================
test_name = "3.2.4.NN_Read_WM"
if rank == 0:
    print(test_name)

# READING
st_time = timeit.default_timer()

# Source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Destination Grid
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

dst_nes = create_nes(comm=None, info=False, projection='lcc', lat_1=lat_1, lat_2=lat_2, lon_0=lon_0, lat_0=lat_0,
                     nx=nx, ny=ny, inc_x=inc_x, inc_y=inc_y, x_0=x_0, y_0=y_0, parallel_method=parallel_method,
                     times=src_nes.get_full_times())
comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()

interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='NN',
                                            weight_matrix_path="NN_WM_NAMEE_to_IP.nc")
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

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
