#!/bin/bash
#SBATCH -n 8 --gres=gpu:volta:1 -o output/pre_bc-0004_vqvae_varClass_rectangle_Cvae32.log-%j


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

ulimit -n # check current open files limit
ulimit -n 65536
ulimit -n
export COLUMNS=200 # increase hydra text length

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
MY_EXPERIMENT="next_scale"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_n_step16"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_L2_traj"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_L2_traj_gradFix"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_L2_traj_gradFix_noXEntropyLoss"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_vqvae_logits_gradFix"
MY_TASK_NAME=$MY_EXPERIMENT"-0003_vqvae_loss_using_logits__loss_sum_fix"
MY_TASK_NAME=$MY_EXPERIMENT"-0003-2_vocabSize4096_Cvae64_quantizeClass"
MY_TASK_NAME=$MY_EXPERIMENT"-0004_vqvae_varClass_rectangle_Cvae32"
# MY_TASK_NAME=$MY_EXPERIMENT"-testing"

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin
# alias torchrun='/home/gridsan/thlin/.conda/envs/catk/bin/torchrun'

# /home/gridsan/thlin/.conda/envs/catk/bin/torchrun \

echo "torchrun -m src.run experiment=$MY_EXPERIMENT task_name=$MY_TASK_NAME"

# https://discuss.pytorch.org/t/training-on-gpus-from-runtimeerror-address-already-in-use-to-timeout/172460

torchrun \
  -m \
  --master-port $MASTER_PORT \
  src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  +config_name=run_scale.yaml  # override @hydra.main(config_path="../configs/", config_name="run.yaml"

  echo "bash train.sh done!"