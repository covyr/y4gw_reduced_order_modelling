#!/bin/bash
#SBATCH --job-name=p4peNS_3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL        # Send email on job start, end and abortion
#SBATCH --mail-user=crc972@student.bham.ac.uk   # Your email address

source activate /rds/projects/p/pratteng-gwastro/conda/envs/igwn-py311-2023

python p4peline.py 'NS' 4