#!/bin/sh
#SBATCH --ntasks-per-node=4 --gres=gpu:volta:1 -o output/local_val.log-%j

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="local_val"
VAL_K=48
MY_TASK_NAME=$MY_EXPERIMENT-K$VAL_K"-debug"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk
source activate catk

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin

# local_val runs on single GPU
python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  model.model_config.validation_rollout_sampling.num_k=$VAL_K \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME

echo "bash local_val.sh done!"