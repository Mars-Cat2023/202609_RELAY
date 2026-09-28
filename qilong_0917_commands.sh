# This file records Bash commands for the Sudoku project; 
# it is documentation, not an executable script.

############# Need to Run Every Time ############
source /data/qilong/miniconda3/etc/profile.d/conda.sh
conda activate relay-sudoku

export PROJECT_ROOT=/data/qilong/202609_RELAY/sudoku
export DATA_DIR=/data/qilong/202609_RELAY/data
export HF_HOME=/data/qilong/202609_RELAY/hf_home
export LOG_DIR=/data/qilong/202609_RELAY/logs
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0
######### ========== END HERE ========== ########

mkdir -p "$DATA_DIR" "$HF_HOME" "$LOG_DIR"


# Smoke Test
xlm \
  job_type=train \
  job_name=sudoku_extreme_smoke \
  experiment=sudoku_extreme_relay_bptt \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=2 \
  per_device_batch_size=8 \
  global_batch_size=8 \
  loggers.wandb=null

# Output Locations:
# /data/qilong/202609_RELAY/
# ├── data/
# ├── hf_home/
# └── logs/
#     └── sudoku_relay_smoke/
#         ├── checkpoints/
#         ├── runs/
#         └── tensorboard/

xlm \
  job_type=train \
  job_name=sudoku_extreme_smoke_2 \
  experiment=sudoku_extreme_mlm_uniform \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=2 \
  per_device_batch_size=8 \
  global_batch_size=8 \
  loggers.wandb=null


#################################################
# Reproducing Table 1
export PROJECT_ROOT="$PWD"
SEED=1   # paper averages seeds 1, 2, 3

# Mask-uniform CE
xlm job_type=train \
  job_name=sudoku_extreme_mlm_uniform_300k_untied_seed$SEED \
  experiment=sudoku_extreme_mlm_uniform \
  seed=$SEED \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=mlm_uniform \
  +tags.seed=$SEED

# Rollout-buffer-only
xlm job_type=train \
  job_name=sudoku_extreme_rollout_300k_untied_seed$SEED \
  experiment=sudoku_extreme_relay_bptt \
  seed=$SEED \
  model=rotary_transformer_xtiny \
  loss.with_relay=false \
  loss.stop_grad_h_s=true \
  predictor.with_relay=false \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=rollout \
  +tags.seed=$SEED

# Relay-sg
xlm job_type=train \
  job_name=sudoku_extreme_relay_sg_300k_untied_seed$SEED \
  experiment=sudoku_extreme_relay_bptt \
  seed=$SEED \
  loss.with_relay=true \
  loss.stop_grad_h_s=true \
  predictor.with_relay=true \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=relay_sg \
  +tags.seed=$SEED

# Relay (BPTT, T=2)
xlm job_type=train \
  job_name=sudoku_extreme_relay_bptt_steps2_300k_untied_seed$SEED \
  experiment=sudoku_extreme_relay_bptt \
  seed=$SEED \
  loss.with_relay=true \
  loss.stop_grad_h_s=false \
  loss.num_steps=2 \
  predictor.with_relay=true \
  ++trainer.precision=bf16-mixed \
  trainer.max_steps=300000 \
  trainer.val_check_interval=5000 \
  per_device_batch_size=512 \
  global_batch_size=512 \
  +tags.sweep=sudoku_extreme_300k \
  +tags.embed_tying=untied \
  +tags.objective=relay \
  +tags.seed=$SEED