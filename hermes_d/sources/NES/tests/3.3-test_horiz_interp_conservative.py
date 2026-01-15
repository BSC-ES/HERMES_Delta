#!/usr/bin/env python
import sys
import os
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf, create_nes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'

result_path = "Times_test_3.3_horiz_interp_conservative.py_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['3.3.1.Only interp', '3.3.2.Create_WM', "3.3.3.Use_WM", "3.3.4.Read_WM"])

src_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/MONARCH_d01_2022111512.nc"
src_type = 'NAMEE'
var_list = ['O3']
# src_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/nox_no_201505.nc"
# src_type = 'CAMS_glob_antv21'
# var_list = ['nox_no']

# ======================================================================================================================
# ======================================        Only interp        =====================================================
# ======================================================================================================================

test_name = '3.3.1.Only interp'
if rank == 0:
    print(test_name)

# READ
# final_dst.variables[var_name]['data'][time, lev] = np.sum(weights * src_aux, axis=1)

st_time = timeit.default_timer()

# Read source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Create destination grid
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
dst_type = "IP"

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()

# INTERPOLATE
interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='Conservative', info=False)
# interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='Conservative', weight_matrix_path='T_WM.nc')
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================         Create_WM        =====================================================
# ======================================================================================================================

test_name = '3.3.2.Create_WM'
if rank == 0:
    print(test_name)

# READING
st_time = timeit.default_timer()

# Read source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Create destination grid
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
dst_type = "IP"

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# Cleaning WM
if os.path.exists("CONS_WM_{0}_to_{1}.nc".format(src_type, dst_type)) and rank == 0:
    os.remove("CONS_WM_{0}_to_{1}.nc".format(src_type, dst_type))
comm.Barrier()

# INTERPOLATE
st_time = timeit.default_timer()
wm_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='Conservative', info=True,
                                        weight_matrix_path="CONS_WM_{0}_to_{1}.nc".format(src_type, dst_type),
                                        only_create_wm=True)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
# st_time = timeit.default_timer()
# interp_nes.to_netcdf(test_name.replace(' ', '_') + ".nc")
# comm.Barrier()
# result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================           Use_WM         =====================================================
# ======================================================================================================================

test_name = "3.3.3.Use_WM"
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()

# Read source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Create destination grid
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
dst_type = "IP"

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# INTERPOLATE
st_time = timeit.default_timer()
interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='Conservative', wm=wm_nes)
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

# ======================================================================================================================
# ======================================           Read_WM         =====================================================
# ======================================================================================================================

test_name = "3.3.4.Read_WM"
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()

# Read source data
src_nes = open_netcdf(src_path, parallel_method=parallel_method)
src_nes.keep_vars(var_list)
src_nes.load()

# Create destination grid
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
dst_type = "IP"

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# INTERPOLATE
st_time = timeit.default_timer()
interp_nes = src_nes.interpolate_horizontal(dst_grid=dst_nes, kind='Conservative',
                                            weight_matrix_path="CONS_WM_{0}_to_{1}.nc".format(src_type, dst_type))
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
st_time = timeit.default_timer()
interp_nes.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size))
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
