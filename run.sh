#!/usr/bin/env bash

set -x

PARTITION=mm_research
JOB_NAME=pretrain
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=16
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=spot \
    ${SRUN_ARGS} \
    python -u main.py pretrain_skeletonclr --config config/CrosSCLR/skeletonclr.yaml
