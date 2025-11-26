#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=5 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanrtx:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=4:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
futhark bench --backend=opencl \
	--json segment-opencl.json \
	segment.fut
futhark bench --backend=c \
	--json segment-c.json \
	segment.fut
futhark bench --backend=multicore \
	--json segment-multicore.json \
	segment.fut
