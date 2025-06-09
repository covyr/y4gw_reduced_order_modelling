#!/bin/bash
#SBATCH --job-name=pipeAS44_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36GB
#SBATCH --time=25:59:00
#SBATCH --mail-type=ALL        # Send email on job start, end and abortion
#SBATCH --mail-user=crc972@student.bham.ac.uk   # Your email address

source activate /rds/projects/p/pratteng-gwastro/conda/envs/igwn-py311-2023

python pipelineAS.py 4 '4,4' 1e-10 1e-10
