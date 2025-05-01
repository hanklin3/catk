#!/bin/bash
#SBATCH -n 8 --gres=gpu:volta:1 -o output/minigrid-016-2_trajPastWithMap2trajFuture.log-%j


export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

# source /etc/profile
# module unload anaconda
# PYTHON=/usr/bin/python3.11
# export PYTHONPATH=/home/gridsan/thlin/.conda/envs/catk/lib/python3.11/site-packages:$PYTHONPATH

# source activate catk

module load anaconda/Python-ML-2024b
source activate var_catk

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin
# alias torchrun='/home/gridsan/thlin/.conda/envs/catk/bin/torchrun'
export PATH=/state/partition1/llgrid/pkg/anaconda/python-ML-2024b/bin:$PATH

# /home/gridsan/thlin/.conda/envs/catk/bin/torchrun \

ulimit -n # check current open files limit
ulimit -n 65536
ulimit -n
export COLUMNS=200 # increase hydra text length

MY_EXPERIMENT="minigrid"

export MASTER_PORT=44144
export MASTER_ADDR=127.0.0.1

while true
do
    status="$(nc -z 127.0.0.1 $MASTER_PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
    echo "MASTER_PORT already in use, retrying...$MASTER_PORT"
    MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
done
echo $MASTER_PORT

################################
# MY_EXPERIMENT="pre_bc"
# MY_EXPERIMENT="clsft"
# MY_EXPERIMENT="minigrid"

# MY_TASK_NAME=$MY_EXPERIMENT"-001_classify_traj"
# MY_TASK_NAME=$MY_EXPERIMENT"-testing"

torchrun \
  -m \
  --master-port $MASTER_PORT \
  src.smart.minigrid.train_vqvae_var \
#   experiment=$MY_EXPERIMENT \
#   task_name=$MY_TASK_NAME
