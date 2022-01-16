#!/bin/bash

# Written by Dánnell Quesada-Chacón

bash run_seeds.sh full 1 1 

bash run_1_seed.sh 1 10 0 30889
bash run_1_seed.sh 1 10 1 30889
bash run_1_seed.sh 1-2 10 1 30889

bash run_1_seed.sh 2 10 0 7777
bash run_1_seed.sh 2 10 1 7777
bash run_1_seed.sh 2-2 10 1 7777

bash run_1_seed.sh 3 10 0 333 
bash run_1_seed.sh 3 10 1 333
bash run_1_seed.sh 3-2 10 1 333

bash run_1_seed.sh 1 10 0 4096
bash run_1_seed.sh 4 10 0 7777 _full # This needs more time
bash run_1_seed.sh 5 10 0 101

