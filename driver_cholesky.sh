#!/bin/bash

# NUM_CPUS=1 NUM_BLOCKS=8    BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=8    BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=8   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=16  BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=32  BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# 
# NUM_CPUS=1 NUM_BLOCKS=8    BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=64   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=8    BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=64   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=8   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=16  BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=32  BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=64  BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# 
# NUM_CPUS=1 NUM_BLOCKS=8     BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=16    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=32    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=1 NUM_BLOCKS=64    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=8     BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=16    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=32    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=4 NUM_BLOCKS=64    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=8    BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh
# NUM_CPUS=16 NUM_BLOCKS=64   BLOCK_SIZE=512 sbatch -c 32 -n 16 run_cholesky.sh

# NUM_CPUS=1  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=2  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=4  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=8  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
# 
# NUM_CPUS=1  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 2 run_cholesky.sh
# NUM_CPUS=2  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 2 run_cholesky.sh
# NUM_CPUS=4  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 2 run_cholesky.sh
# NUM_CPUS=8  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 2 run_cholesky.sh
# 
# NUM_CPUS=1  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=2  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=4  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# NUM_CPUS=8  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 4 run_cholesky.sh
# 
# NUM_CPUS=1  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=2  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=4  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=8  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh

##### Block size 256

# NUM_CPUS=2   NUM_BLOCKS=32   BLOCK_SIZE=256 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=2   NUM_BLOCKS=64   BLOCK_SIZE=256 sbatch -c 32 -n 1 run_cholesky.sh
# 
# NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=256 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=256 sbatch -c 32 -n 1 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=128  BLOCK_SIZE=256 sbatch -c 32 -n 1 run_cholesky.sh
# 
# NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=256 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=256 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=128  BLOCK_SIZE=256 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=256  BLOCK_SIZE=256 sbatch -c 32 -n 8 run_cholesky.sh
# 
# NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=256 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=256 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=128  BLOCK_SIZE=256 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=256  BLOCK_SIZE=256 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=512  BLOCK_SIZE=256 sbatch -c 32 -n 64 run_cholesky.sh

##### Block size 512

NUM_CPUS=2   NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
NUM_CPUS=2   NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh

NUM_CPUS=16  NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh
NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=512 sbatch -c 32 -n 1 run_cholesky.sh

NUM_CPUS=16  NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh
NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh
NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh
#NUM_CPUS=16  NUM_BLOCKS=128  BLOCK_SIZE=512 sbatch -c 32 -n 8 run_cholesky.sh

NUM_CPUS=16  NUM_BLOCKS=16   BLOCK_SIZE=512 sbatch -c 32 -n 64 run_cholesky.sh
NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=512 sbatch -c 32 -n 64 run_cholesky.sh
NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=512 sbatch -c 32 -n 64 run_cholesky.sh
#NUM_CPUS=16  NUM_BLOCKS=128  BLOCK_SIZE=512 sbatch -c 32 -n 64 run_cholesky.sh
#NUM_CPUS=16  NUM_BLOCKS=256  BLOCK_SIZE=512 sbatch -c 32 -n 64 run_cholesky.sh

##### Block size 1024

# NUM_CPUS=16  NUM_BLOCKS=8    BLOCK_SIZE=1024 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=16   BLOCK_SIZE=1024 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=1024 sbatch -c 32 -n 8 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=1024 sbatch -c 32 -n 8 run_cholesky.sh
# 
# NUM_CPUS=16  NUM_BLOCKS=8    BLOCK_SIZE=1024 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=16   BLOCK_SIZE=1024 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=32   BLOCK_SIZE=1024 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=64   BLOCK_SIZE=1024 sbatch -c 32 -n 64 run_cholesky.sh
# NUM_CPUS=16  NUM_BLOCKS=128  BLOCK_SIZE=1024 sbatch -c 32 -n 64 run_cholesky.sh
