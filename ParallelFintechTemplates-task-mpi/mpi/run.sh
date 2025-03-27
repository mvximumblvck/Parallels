#!/bin/bash
#SBATCH --comment="Test MPI"
#SBATCH --ntasks=4
#SBATCH --partition=RT_study
#SBATCH --error=error.log
#SBATCH --output=output.log
#SBATCH --mem=4G

mpiexec --mca btl tcp,vader,self --mca btl_tcp_if_include eno1 python main.py --mode parallel
