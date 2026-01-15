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

result_path = "Times_test_2.5_longitudes_{0}_{1:03d}.csv".format(parallel_method, size)
result = pd.DataFrame(index=['read', 'calculate', 'write'],
                      columns=['2.5.1.Longitude_conversion'])

# ======================================================================================================================
# =====================================  FILE WITH LONGITUDES in [0, 360]  =============================================
# ======================================================================================================================

test_name = "2.5.1.Longitude_conversion"
if rank == 0:
    print(test_name)

# READ
st_time = timeit.default_timer()

# NC file with longitudes in [0, 360].
path_1 = '/gpfs/projects/bsc32/models/NES_tutorial_data/preprocessed_backup.nc'
nessy_1 = open_netcdf(path=path_1, parallel_method=parallel_method, info=True)

comm.Barrier()
result.loc['read', test_name] = timeit.default_timer() - st_time

# CONVERT LONGITUDES
st_time = timeit.default_timer()
nessy_1.load()
print('Rank', rank, '-', 'Convert Longitudes', nessy_1.convert_longitudes())
comm.Barrier()
result.loc['calculate', test_name] = timeit.default_timer() - st_time

# WRITE
path_2 = test_name.replace(' ', '_') + "_{0:03d}.nc".format(size)
st_time = timeit.default_timer()
nessy_1.to_netcdf(path_2, info=True)
comm.Barrier()
result.loc['write', test_name] = timeit.default_timer() - st_time

comm.Barrier()
if rank == 0:
    print(result.loc[:, test_name])
sys.stdout.flush()

if rank == 0:
    result.to_csv(result_path)
    print("TEST PASSED SUCCESSFULLY!!!!!")
