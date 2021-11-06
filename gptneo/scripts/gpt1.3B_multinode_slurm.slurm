#!/bin/bash

#SBATCH -J GPTNEO_16nodes                  # Job name
#SBATCH -o slurm_logs/gpt1.3B_16nodes.o%j   # Name of stdout output file
#SBATCH -e slurm_logs/gpt1.3B_16nodes.e%j   # Name of stdout output file
#SBATCH -N 16                              # Total # of nodes 
#SBATCH -n 64                             # Total # of mpi tasks
#SBATCH -t 48:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=XXX                   # Email address
#SBATCH --mail-type=end                   # Send email at begin and end of job
#SBATCH -p rtx                            # Queue

# Note mpiexec calls may need to be changed depending on MPI type

HOSTFILE=/tmp/hostfile

scontrol show hostnames $SLURM_NODELIST > $HOSTFILE
cat $HOSTFILE

GPU_PER_NODE=4
NODES=$(wc -l < $HOSTFILE)
MASTER_NODE=$(head -n 1 $HOSTFILE)

mpiexec.hydra -f $HOSTFILE -np $NODES -ppn 1 \
  ./scripts/gpt1.3B_multinode_torch.sh \
    --ngpus $GPU_PER_NODE --nnodes $NODES --master $MASTER_NODE
