#!/bin/bash
#SBATCH -o legion_chol_%j.out
#SBATCH -t 00:15:00

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname

let MATRIX_SIZE=BLOCK_SIZE*NUM_BLOCKS
export LD_LIBRARY_PATH="$(pwd):$LD_LIBRARY_PATH"

echo ">>>>slurm_id=${SLURM_JOB_ID},matrix_size=${MATRIX_SIZE},num_blocks=${NUM_BLOCKS},block_size=${BLOCK_SIZE},num_ranks=${SLURM_NTASKS},num_cpus=${NUM_CPUS}"

# LAUNCHER="mpirun -n ${SLURM_NTASKS}" ./regent.py exam#ples/my_cholesky.rg -fflow 0 -level 5 -n ${MATRIX_SIZE} -p ${NUM_BLOCKS} -ll:csize 16384 -foverride-demand-index-launch 1 -ll:cpu ${NUM_CPUS} -ll:util ${NUM_CPUS} -lg:prof ${SLURM_NTASKS} -lg:prof_logfile prof_${SLURM_JOB_ID}_%.gz
LAUNCHER="mpirun -n ${SLURM_NTASKS}" ~/legion/language/regent.py ./my_cholesky.rg -fflow 0 -level 5 -n ${MATRIX_SIZE} -p ${NUM_BLOCKS} -ll:csize 16384 -foverride-demand-index-launch 1 -ll:cpu ${NUM_CPUS} -ll:util ${NUM_CPUS}
