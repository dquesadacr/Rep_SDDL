#!/bin/bash

# Written by Dánnell Quesada-Chacón

cd /data
pwd
echo "Command to run: PYTHONHASHSEED=$4 TF_DETERMINISTIC_OPS=$3 Rscript OEG_cnn_$1.R $2"

PYTHONHASHSEED=$4 TF_DETERMINISTIC_OPS=$3 Rscript OEG_cnn_$1.R $2
