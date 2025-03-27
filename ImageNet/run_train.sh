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
MODEL="swin_v2_b"

srun --mpibind=off python -u train.py \
                        --world-size 16 \
                        --corrupt-prob 0.2 \   #Alpha 
                        --model $MODEL \
                        --epochs 400 \
                        --batch-size 32 \
                        --opt adamw \
                        --lr 0.0001 \
                        --weight-decay 0.05 \
                        --norm-weight-decay 0.0  \
                        --bias-weight-decay 0.0 \
                        --transformer-embedding-decay 0.0 \
                        --lr-scheduler cosineannealinglr \
                        --lr-min 0.000001 \
                        --amp \
                        --label-smoothing 0.1 \
                        --mixup-alpha 0.8 \
                        --clip-grad-norm 5.0 \
                        --cutmix-alpha 1.0 \
                        --random-erase 0.25 \
                        --interpolation bicubic \
                        --auto-augment ta_wide \
                        --model-ema \
                        --ra-sampler \
                        --ra-reps 4  \
                        --val-resize-size 256 \
                        --val-crop-size 256 \
                        --train-crop-size 256 \
