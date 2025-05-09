#!/bin/bash
#SBATCH --job-name=BM
#SBATCH -o ./log/BM.%A_%a.out
#SBATCH -e ./log/BM.%A_%a.out
#SBATCH --partition=cpu
#\\SBATCH --nodelist=n06
#SBATCH --exclude=n05
#SBATCH -N 1
#SBATCH -n 2
#\\SBATCH --mem=10G
#SBATCH --array=0-199%20


i=$SLURM_ARRAY_TASK_ID
echo ${i}
python E2_BrainsmashForEigenmodes.py --modei ${i}