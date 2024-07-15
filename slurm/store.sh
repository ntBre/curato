#!/bin/bash
#SBATCH -J store
#SBATCH -p standard
#SBATCH -t 144:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=140gb
#SBATCH --account dmobley_lab
#SBATCH --export ALL
#SBATCH --mail-user=bwestbr1@uci.edu
#SBATCH --constraint=fastscratch
#SBATCH -o logs/store.out

source $HOME/.bashrc

conda activate fb-196-qcnew

python store.py -n $SLURM_CPUS_PER_TASK
