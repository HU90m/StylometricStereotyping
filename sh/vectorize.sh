#!/bin/bash

#SBATCH --mem-per-cpu=64000

cd $HOME/stylometricStereotyping/python

module load python/3.7.3

python3 vectorize.py /scratch/hm6g17/data/training-csv /scratch/hm6g17/data/vectors > $HOME/stdout_files/vec.txt 2>&1
