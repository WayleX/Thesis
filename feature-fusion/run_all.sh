#!/bin/bash
# Run all 26 thesis experiments: train + evaluate
# Usage: bash run_all.sh [GPU_ID]

GPU=${1:-0}
EPOCHS=25

echo "=== Running all experiments on GPU $GPU ==="

for cfg in configs/*.yaml; do
    name=$(basename "$cfg" .yaml)
    echo ""
    echo "=========================================="
    echo "  Training: $name"
    echo "=========================================="
    python train_exp.py --config "$cfg" --gpu "$GPU" --epochs "$EPOCHS"

    echo ""
    echo "  Evaluating: $name"
    echo "----------------------------------------"
    python evaluate.py --config "$cfg" --gpu "$GPU"
done

echo ""
echo "=== All experiments complete ==="
echo "Results saved to results/"
echo ""
echo "Generating analysis figures..."
python analysis.py --results-dir results/ --output-dir figures/
python full_analysis.py
echo "Done."
