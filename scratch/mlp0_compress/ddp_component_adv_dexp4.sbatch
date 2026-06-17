#!/bin/bash
#SBATCH --job-name=comp-mlp-adv-dexp4-ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --comment="Disentangled component MLP, d_expand=4, 8-GPU DDP, effective batch 256. 2-term loss 0.5*relmse(stochastic) + 0.5*relmse(persistent adversarial, per_batch_per_position PPGD). s-55ea3f9b; oli/opus"
#SBATCH --output=/mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress/scratch/mlp0_compress/ddp_component_adv_dexp4_%j.log

# Tuning config for the adversarial ComponentMLP run. Edit the values below and resubmit
# with `sbatch ddp_component_adv_dexp4.sbatch`. Every argument is a `main()` parameter in
# run_component.py (fire CLI) — anything not listed here falls back to its default there.
#
# Parallelism: NPROC GPUs via DDP on one node. Per-rank batch is BATCH_SIZE sequences of
# length 512; effective batch = NPROC * BATCH_SIZE. Model grads are all-reduced each step;
# the PPGD adversary sources are per-rank (per_batch_per_position is local to each rank).

NPROC=8

# --- target / architecture ---
D_EXPAND=4              # GeLU neurons per output subcomponent (0 = bypass-only)

# --- optimisation (the replacement MLPs) ---
STEPS=100000
BATCH_SIZE=32           # per-rank; effective batch = NPROC * BATCH_SIZE
LR=1e-4                 # base lr, cosine schedule
WARMUP_STEPS=200
FINAL_LR_FRAC=0.1

# --- loss term coefficients ---
STOCHASTIC_COEFF=0.5    # relmse under one stochastic mask draw
ADVERSARIAL_COEFF=0.5   # relmse under persistent adversarial (PPGD) masks; 0 disables

# --- persistent adversary (PPGD ascent on the mask sources) ---
ADV_LR=0.01
ADV_BETA1=0.0
ADV_BETA2=0.99
ADV_N_WARMUP=1          # extra source updates/step (total = ADV_N_WARMUP + 1)
ADV_SCOPE=per_batch_per_position   # or shared_across_batch

# --- eval-time PGD adversary (fresh sources each eval, shared across batch) ---
PGD_STEP_SIZE=0.1
PGD_N_STEPS=20

# --- cadence ---
EVAL_EVERY=1000
N_EVAL_BATCHES=4
SAVE_EVERY=10000

cd /mnt/polished-lake/home/oli/opus-param-decomp/wt-mlp0-compress
source .venv/bin/activate
cd scratch/mlp0_compress
torchrun --standalone --nproc_per_node=$NPROC run_component.py \
  --d_expand $D_EXPAND --steps $STEPS --batch_size $BATCH_SIZE \
  --lr $LR --warmup_steps $WARMUP_STEPS --final_lr_frac $FINAL_LR_FRAC \
  --stochastic_coeff $STOCHASTIC_COEFF --adversarial_coeff $ADVERSARIAL_COEFF \
  --adv_lr $ADV_LR --adv_beta1 $ADV_BETA1 --adv_beta2 $ADV_BETA2 \
  --adv_n_warmup $ADV_N_WARMUP --adv_scope $ADV_SCOPE \
  --pgd_step_size $PGD_STEP_SIZE --pgd_n_steps $PGD_N_STEPS \
  --eval_every $EVAL_EVERY --n_eval_batches $N_EVAL_BATCHES --save_every $SAVE_EVERY
