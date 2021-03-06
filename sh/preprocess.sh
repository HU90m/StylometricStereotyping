#!/bin/bash

#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1

cd $HOME/stylometricStereotyping/python

module load python/3.7.3

python3 preprocess.py /scratch/hm6g17/data/training/en /scratch/hm6g17/data/training_csv > $HOME/stdout_files/preprocess.txt 2>&1
