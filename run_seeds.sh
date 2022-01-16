#!/bin/bash

# Written by Dánnell Quesada-Chacón

mkdir -p jobs

for i in $(cat seeds); do echo CNN_file=$1 n_repeat=$2 deterministic=$3 seed=$i; sbatch -J $1-$2-$3-$i -o $(pwd)/jobs/cnn_"$1"_d-"$3"_s-"$i".out -e $(pwd)/jobs/cnn_"$1"_d-"$3"_s-"$i".err job_alphacentauri_full.sh $1 $2 $3 $i; done

