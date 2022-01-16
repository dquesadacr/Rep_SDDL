#!/bin/bash

# Written by Dánnell Quesada-Chacón

mkdir -p jobs

echo CNN_file=$1 n_repeat=$2 deterministic=$3 seed=$4 job_file=job_alphacentauri"$5".sh
sbatch -J $1-$2-$3-$4 -o $(pwd)/jobs/cnn_"$1"_d-"$3"_s-"$4".out -e $(pwd)/jobs/cnn_"$1"_d-"$3"_s-"$4".err job_alphacentauri"$5".sh $1 $2 $3 $4
