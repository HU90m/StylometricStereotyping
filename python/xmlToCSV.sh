#!/bin/bash

#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1

cd $HOME/stylometricStereotyping/python

module load python/3.7.3

python3 xmlToCSV.py /scratch/hm6g17/data/training-csv-2 /scratch/hm6g17/data/training/en/
