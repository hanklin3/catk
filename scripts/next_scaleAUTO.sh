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

module load anaconda/Python-ML-2024b
source activate var_catk

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin
# alias torchrun='/home/gridsan/thlin/.conda/envs/catk/bin/torchrun'
# export PATH=/state/partition1/llgrid/pkg/anaconda/python-ML-2024b/bin:$PATH

# /home/gridsan/thlin/.conda/envs/catk/bin/torchrun \

ulimit -n # check current open files limit
ulimit -n 65536
ulimit -n
export COLUMNS=200 # increase hydra text length

MY_EXPERIMENT="next_scale_autoreg"
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

torchrun \
  -m \
  --master-port $MASTER_PORT \
  src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  # +config_name=run_scale_autoreg.yaml  # override @hydra.main(config_path="../configs/", config_name="run.yaml"