#!/bin/bash 

#SBATCH --job-name=aenc
#SBATCH --partition=rome
#SBATCH --output=my_output.stdout
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=8:00:00

module load gcc/13.2.0-iqpfkya cmake/3.27.9-nmh6tto eigen/3.4.0-titj7ys 

MY_HOME="/home/c/cioakeim"
MY_HOME="/home/chris"

project_dir="/home/c/cioakeim/nns/neural-networks-hw3"
project_dir="/home/chris/Documents/programms/nns/neural-networks-hw3"

source $MY_HOME/aocl/5.0.0/aocc/amd-libs.cfg

config_path="$project_dir/jobs/$1"

cd "$project_dir"
mkdir -p build
cd build
#cmake -DMY_HOME_DIR="$MY_HOME" -DMKL_INTERFACE_FULL=intel_lp64 ..
cmake -DHOME_DIR=$MY_HOME ..
make


./autoEncoderTest $config_path
