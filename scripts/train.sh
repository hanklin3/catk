#!/bin/bash
#SBATCH -n 8 --gres=gpu:volta:1 -o output/pre_bc_test.log-%j


export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="pre_bc"
# MY_EXPERIMENT="clsft"
MY_TASK_NAME=$MY_EXPERIMENT"-testing"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

# source /etc/profile
# module unload anaconda
# PYTHON=/usr/bin/python3.11
# export PYTHONPATH=/home/gridsan/thlin/.conda/envs/catk/lib/python3.11/site-packages:$PYTHONPATH

# source activate catk

source activate var_catk
module load anaconda/Python-ML-2024b

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin
# alias torchrun='/home/gridsan/thlin/.conda/envs/catk/bin/torchrun'

# /home/gridsan/thlin/.conda/envs/catk/bin/torchrun \

ulimit -n # check current open files limit
ulimit -n 65536
ulimit -n
export COLUMNS=200 # increase hydra text length

torchrun \
  -m \
  src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME

# python3.11 -m torch.distributed.run \
#   src/run.py \
#   experiment=$MY_EXPERIMENT \
#   task_name=$MY_TASK_NAME

# ! below is for training with ddp
# torchrun \
#   --rdzv_id $SLURM_JOB_ID \
#   --rdzv_backend c10d \
#   --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#   --nnodes $NUM_NODES \
#   --nproc_per_node gpu \
#   -m src.run \
#   experiment=$MY_EXPERIMENT \
#   trainer=ddp \
#   task_name=$MY_TASK_NAME

echo "bash train.sh done!"

# https://discuss.pytorch.org/t/training-on-gpus-from-runtimeerror-address-already-in-use-to-timeout/172460