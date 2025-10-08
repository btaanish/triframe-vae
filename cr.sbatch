#!/bin/bash
#SBATCH --job-name=triplane_vae_softnet
#SBATCH --output=logs2/triplane_vae_softnet_%j.out
#SBATCH --error=logs2/triplane_vae_softnet_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=28:00:00

echo "[$(date)] Job starting on: $(hostname)"

# --- Activate environment ---
source ~/venvs/llama2-70b/bin/activate
export PYTHONPATH=~/venvs/llama2-70b/lib/python3.12/site-packages:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
which python
python --version
python -m site

# --- Run training under srun with full env path ---
srun ~/venvs/llama2-70b/bin/python ~/SoftNet/Soft-Fusion-FYP/train_stage1_triplane_vae.py \
    --name triplane_vae_softnet \
    --dataset_mode softnet_sdf \
    --dataroot "data/Softnet/SDF_v1/resolution_128" \
    --cat all \
    --res 128 \
    --batch_size 2 \
    --lr 0.0001 \
    --total_iters 200000 \
    --save_steps_freq 10000 \
    --print_freq 100 \
    --display_freq 1000 \
    --gpu_ids 0\

echo "[$(date)] Job finished."
