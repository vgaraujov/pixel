#!/bin/bash

#SBATCH -J Inf_Loc
#SBATCH -o /home/david.mogrovejo/PVLM/pixel/result_dir/vit_training_1.log
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230GB 

nvidia-smi

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 train_vilt.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_vilt_accelerate.py

nvidia-smi
