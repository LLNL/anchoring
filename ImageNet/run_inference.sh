#!/bin/sh
#SBATCH -N 8
#SBATCH -A ams
#SBATCH --partition=pvis
#SBATCH --ntasks-per-node=2
#SBATCH -t 11:59:00
#SBATCH --export=ALL

export MASTER_ADDR=`scontrol show hostname ${SLURM_NODELIST} | head -n1`
export MASTER_PORT=23456
export WORLD_SIZE=16
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CUDA_VISIBLE_DEVICES=0,1

echo "NODELIST="${SLURM_NODELIST}
echo "MASTER_ADDR="$MASTER_ADDR

source ~/.bashrc  

dset_list=("imagenet-c"
           "imagenet-cbar"
           "imagenet-r"
           "imagenet-sketch")

dset_path_list=("../data/imagenet-c/imagenet-c"
                "../data/imagenet-cbar/"
                "../data/imagenet-r"
                "../data/imagenet-sketch")

# Iterate through the list
for i in "${!dset_list[@]}"
do
    echo "${dset_path_list[i]}"
    ckpt="swin_v2_b-781e5279" # Specify Checkpoint name
    model="swin_v2_b"
    srun --mpibind=off python -u run_corruptions.py --dset "${dset_list[i]}"  --data-path "${dset_path_list[i]}" --world-size 16 --batch-size 64 --model $model --ckpt_name $ckpt --output-dir logs --interpolation bicubic --val-resize-size 272 --val-crop-size 256 --train-crop-size 256
    
    ckpt="vanilla_duq_372_top1_84.06" # Specify Checkpoint name
    model="anchored_swin_v2_b"
    srun --mpibind=off python -u run_corruptions.py --dset "${dset_list[i]}"  --data-path "${dset_path_list[i]}" --world-size 16 --batch-size 64 --model $model --ckpt_name $ckpt --output-dir logs --interpolation bicubic --val-resize-size 256 --val-crop-size 256 --train-crop-size 256
    
    ckpt="zero_crop_p_0.2_373_top1_84.09" # Specify Checkpoint name
    model="anchored_swin_v2_b"
    srun --mpibind=off python -u run_corruptions.py --dset "${dset_list[i]}"  --data-path "${dset_path_list[i]}" --world-size 16 --batch-size 64 --model $model --ckpt_name $ckpt --output-dir logs --interpolation bicubic --val-resize-size 256 --val-crop-size 256 --train-crop-size 256
    
done