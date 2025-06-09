#!/bin/bash
#SBATCH --job-name=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL        # Send email on job start, end and abortion
#SBATCH --mail-user=crc972@student.bham.ac.uk   # Your email address

source activate ../../../../../projects/p/pratteng-gwastro/conda/envs/igwn-py311-2023

python p1peline.py 3 '2,2'