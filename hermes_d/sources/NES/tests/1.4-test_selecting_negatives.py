#!/usr/bin/env python
import sys
import timeit
import pandas as pd
from mpi4py import MPI
from nes import open_netcdf
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parallel_method = 'Y'
serial_write = True

result_path = "Times_test_1.4.Selecting_{0}_{1:03d}.csv".format(parallel_method, size)

result = pd.DataFrame(index=['read', 'calcul', 'write'],
                      columns=['1.4.1.Selecting_Negatives'])

# Global GFAS
src_path = "/gpfs/projects/bsc32/models/NES_tutorial_data/ga_20220101.nc"
var_list = ['so2', "pm25", 'nox']

# ======================================================================================================================
# ======================================      '1.4.1.Selecting_Negatives'       ========================================
# ======================================================================================================================
test_name = '1.4.1.Selecting_Negatives'

if rank == 0:
    print(test_name)

st_time = timeit.default_timer()

# Source data
nessy = open_netcdf(src_path, parallel_method=parallel_method, balanced=True)
nessy.keep_vars(var_list)
nessy.sel(lat_min=35, lat_max=45, lon_min=-9, lon_max=5)
nessy.cell_measures = {}
# nessy.calculate_grid_area(overwrite=True)
nessy.load()
print(f"Rank {nessy.rank} -> Lon: {nessy.lon}")
print(f"Rank {nessy.rank} -> Lon BNDS: {nessy.lon_bnds}")
print(f"Rank {nessy.rank} -> FULL Lon: {nessy.get_full_longitudes()}")

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

st_time = timeit.default_timer()
nessy.to_netcdf(test_name.replace(' ', '_') + "{0:03d}.nc".format(size), serial=serial_write)

comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
