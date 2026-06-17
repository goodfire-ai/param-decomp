#!/bin/bash
#SBATCH --job-name=mlp0-compress
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --comment="compress block-0 MLP of s-55ea3f9b to 1/10 neuron dim (KL distill on CI-masked forwards), oli/opus"
#SBATCH --output=/mnt/home/lucius/spd/scratch/mlp_compress/train_%j.log

cd /mnt/home/lucius/spd
source .venv/bin/activate
python scratch/mlp_compress/run.py --steps 20000 --batch_size 32 --lr 1e-3
