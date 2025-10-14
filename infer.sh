#! /bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0 #$gpu_ids
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export PYTHONPATH=`pwd`:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export TEXTLESS_CHECKPOINT_ROOT=ckpt/
# export PYTHONPATH=textlesslib/:fairseq/:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

python -W ignore inference.py \
    --prompt_wav assets/src/121_127105_000009_000000.wav \
    --prompt_text "We say, of course, somebody exclaimed, that they give two turns!" \
    --gen_text "How are you? This is lightspeed studio from tencent." 