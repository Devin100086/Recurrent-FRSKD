#!/bin/bash
# Run a Python script from Bash

python main.py\
        --model cifarresnet18\
        --data_dir data/ \
        --data TINY \
        --batch_size 64 \
        --alpha 3 \
        --beta  100 \
        --temperature 3 \
        --use_distill True \
        --aux none \
        --aux_lamb 0 \
        --aug none \
        --aug_a 0 \
        --gpu_id 0 \