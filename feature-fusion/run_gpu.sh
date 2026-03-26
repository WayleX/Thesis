#!/bin/bash
# Run a set of configs sequentially on a single GPU, then evaluate each
GPU=$1
shift

for cfg in "$@"; do
    name=$(basename "$cfg" .yaml)
    echo "[GPU$GPU] Training: $name"
    python -u train_exp.py --config "$cfg" --gpu "$GPU" --epochs 25 2>&1 | tee "results/${name}_train.log"
    echo "[GPU$GPU] Evaluating: $name"
    python -u evaluate.py --config "$cfg" --gpu "$GPU" --save 2>&1 | tee "results/${name}_eval.log"
    echo "[GPU$GPU] Done: $name"
done
