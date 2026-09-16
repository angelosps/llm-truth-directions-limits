# Testing the Limits of Truth Directions in LLMs

Code for the paper [*Testing the Limits of Truth Directions in LLMs*](https://arxiv.org/abs/2604.03754), BlackboxNLP 2026.

## Setup

```bash
pip install -r requirements.txt
```

Requires access to `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace.

## Reproducing results

The full pipeline has three steps: (1) collect model activations, (2) train linear probes, and (3) generate main paper figures.

### Step 1 & 2: Activation collection and probe training

A script is provided for running both steps for all 9 datasets and 4 instruction settings (no-prompt, ask-correct, ask-tf, ask-able):

```bash
./run_all.sh
```

This will populate the `activation_output/` and `probe_output/` directories.

Alternatively, you can run for each dataset individually as:

```bash
# Collect activations
cd activation_collection
python run.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --instruction no-prompt \
    --dataset_path ../datasets/factual/cities.csv \
    --save_dir ../activation_output \
    --token_mode last

# Train probes
cd ../probe_training
python train_LR_probe.py \
    --correct_path ../activation_output/cities_no-prompt_Llama-3.1-8B-Instruct_last_run_correct.pkl \
    --incorrect_path ../activation_output/cities_no-prompt_Llama-3.1-8B-Instruct_last_run_incorrect.pkl \
    --out_dir ../probe_output \
    --run_tag cities_no-prompt_Llama-3.1-8B-Instruct \
    --balance
```

Available instruction settings: `no-prompt`, `ask-tf`, `ask-able`, `ask-arith`, `ask-correct`, `random-prompt`, `read-prompt`.

### Step 3: Generate figures

Once activations and probes are generated, open and run `reproduce_main_figures.ipynb`. You only need to update `PROBES_DIR` and `ACTS_DIR` paths to point to the directories of saved probes and activations accordingly.

## Computational requirements

- **GPU**: For loading Llama-3.1-8B-Instruct in float16 and running inference.
- **Disk**: ~25 GB for storing activations across all datasets and instructions, and probe outputs.
- **Time**: It took us ~4 hours on a single A100 for running the full pipeline (all 36 runs). 

*The code has been cleaned up using Claude Code (Opus 5).