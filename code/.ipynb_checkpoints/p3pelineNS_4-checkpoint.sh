#!/bin/bash
#SBATCH --job-name=p3peNS_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL        # Send email on job start, end and abortion
#SBATCH --mail-user=crc972@student.bham.ac.uk   # Your email address

source activate /rds/projects/p/pratteng-gwastro/conda/envs/igwn-py311-2023

python p3peline.py 'NS' 4