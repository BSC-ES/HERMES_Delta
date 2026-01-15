#!/bin/bash

EXPORTPATH="/gpfs/scratch/bsc32/bsc32538/NES_tests/NES"
SRCPATH="/gpfs/scratch/bsc32/bsc32538/NES_tests/NES/tests"

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load NES/1.1.3-nord3-v2-foss-2019b-Python-3.7.4


for EXE in "1.1-test_read_write_projection.py" "1.2-test_create_projection.py" "1.3-test_selecting.py" "2.1-test_spatial_join.py" "2.2-test_create_shapefile.py" "2.3-test_bounds.py" "2.4-test_cell_area.py" "2.5-test_longitude_conversion.py" "3.1-test_vertical_interp.py" "3.2-test_horiz_interp_bilinear.py" "3.3-test_horiz_interp_conservative.py" "4.1-test_stats.py" "4.2-test_sum.py" "4.3-test_write_timestep.py"
    do
        for nprocs in 1 2 4 8 16
            do
                JOB_ID=`sbatch --ntasks=${nprocs} --qos=debug --exclusive --job-name=NES_${EXE}_${nprocs} --output=./log_NES-${EXE}_nord3v2_${nprocs}_%J.out --error=./log_NES-${EXE}_nord3v2_${nprocs}_%J.err -D . --time=02:00:00 --wrap="export PYTHONPATH=${EXPORTPATH}:${PYTHONPATH}; cd ${SRCPATH}; mpirun --mca mpi_warn_on_fork 0 -np ${nprocs} python ${SRCPATH}/${EXE}"`
            done
    done
