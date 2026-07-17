#!/bin/bash
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG="Llama-3.1-8B-Instruct"
INSTRUCTIONS=("no-prompt" "ask-correct" "ask-tf" "ask-able")

ARITH_DATASETS=("arith_1op" "arith_2ops" "arith_3ops")
FACT_DATASETS=(
    "cities" "neg_cities" "cities_conj"
    "cities_same_country_quant"
    "cities_exact_k" "cities_exact_k1_k2"
)

ROOT="$(cd "$(dirname "$0")" && pwd)"
ACT_DIR="$ROOT/activation_output"
PROBE_DIR="$ROOT/probe_output"

mkdir -p "$ACT_DIR" "$PROBE_DIR"

for INSTR in "${INSTRUCTIONS[@]}"; do
    echo "=== Instruction: $INSTR ==="

    for d in "${ARITH_DATASETS[@]}"; do
        TAG="${d}_${INSTR}_${MODEL_TAG}"
        echo "  $TAG"
        cd "$ROOT/activation_collection"
        python run.py --model "$MODEL" --instruction "$INSTR" \
            --dataset_path "$ROOT/datasets/arithmetic/${d}.csv" \
            --save_dir "$ACT_DIR" --token_mode last
        cd "$ROOT/probe_training"
        python train_LR_probe.py \
            --correct_path "$ACT_DIR/${TAG}_last_run_correct.pkl" \
            --incorrect_path "$ACT_DIR/${TAG}_last_run_incorrect.pkl" \
            --out_dir "$PROBE_DIR" --run_tag "$TAG" --balance
    done

    for d in "${FACT_DATASETS[@]}"; do
        TAG="${d}_${INSTR}_${MODEL_TAG}"
        echo "  $TAG"
        cd "$ROOT/activation_collection"
        python run.py --model "$MODEL" --instruction "$INSTR" \
            --dataset_path "$ROOT/datasets/factual/${d}.csv" \
            --save_dir "$ACT_DIR" --token_mode last
        cd "$ROOT/probe_training"
        python train_LR_probe.py \
            --correct_path "$ACT_DIR/${TAG}_last_run_correct.pkl" \
            --incorrect_path "$ACT_DIR/${TAG}_last_run_incorrect.pkl" \
            --out_dir "$PROBE_DIR" --run_tag "$TAG" --balance
    done
done

echo "All runs complete."
