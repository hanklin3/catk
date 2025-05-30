#!/bin/bash
#SBATCH -n 8 --gres=gpu:volta:1 -o output/clsft-test.log-%j


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

source activate var_catk
module load anaconda/Python-ML-2024b

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin
# alias torchrun='/home/gridsan/thlin/.conda/envs/catk/bin/torchrun'

# /home/gridsan/thlin/.conda/envs/catk/bin/torchrun \

ulimit -n # check current open files limit
ulimit -n 65536
ulimit -n
export COLUMNS=200 # increase hydra text length

MY_EXPERIMENT="next_scale"
MY_TASK_NAME=$MY_EXPERIMENT"-testCode"

export MASTER_PORT=44144
export MASTER_ADDR=127.0.0.1

# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT
# export MASTER_PORT=$PORT

NUM_NODES=1

CUDA_VISIBLE_DEVICES=3 torchrun \
  -m \
  --master-port $MASTER_PORT \
  --nnodes $NUM_NODES \
  --nproc_per_node 1 \
  src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  +config_name=run_scale.yaml  # override @hydra.main(config_path="../configs/", config_name="run.yaml"