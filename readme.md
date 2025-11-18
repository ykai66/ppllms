# PPllms

PPLlms is a two-stage pruning strategy for large language models. Code for paper: "PP-LLMs: A Progressive Pruning Approach with Medium-Granularity for Large Language Models"

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the pruning script with default parameters:

```bash
python ppllms.py
```

### Advanced Usage

You can customize the pruning process by passing parameters:

```bash
python ppllms.py \
    --model_name /path/to/model \
    --num_blocks 40 \
    --prune_rate 0.25 \
    --mha_remove_layers 10 \
    --ffn_remove_layers 10 \
    --nsamples 128 \
    --dataset /path/to/dataset \
    --eval_ppl True
```

### Automatic Layer Calculation

If you don't specify `mha_remove_layers` and `ffn_remove_layers`, the script will automatically calculate them based on the target prune rate and parameter ratio:

```bash
python ppllms.py \
    --model_name /path/to/model \
    --prune_rate 0.25 \
    --mha_ffn_ratio 2.0
```

## Parameters

### Required Parameters

- `model_name`: Path to the model (default: `/workspace/models/meta-llama/Llama-2-7b-hf`)
- `num_blocks`: Number of blocks in the model (default: `32`)

### Pruning Parameters

- `prune_rate`: Target pruning rate (default: `0.25`, i.e., 25%)
- `mha_remove_layers`: Number of MHA layers to remove (default: `None`, auto-calculated)
- `ffn_remove_layers`: Number of FFN layers to remove (default: `None`, auto-calculated)
- `mha_ffn_ratio`: FFN/MHA parameter ratio for auto-calculation (default: `2.0`)

### Data Parameters

- `nsamples`: Number of samples for evaluation (default: `128`)
- `dataset`: Path to the dataset (default: `/workspace/datasets/allenai/c4`)
- `seed`: Random seed (default: `0`)

### Evaluation Parameters

- `eval_ppl`: Whether to evaluate perplexity (default: `True`)

## Two-Stage Pruning Strategy

### Stage 1: MHA Candidate Selection

1. Calculate Block Importance (BI) scores for all layers
2. Select candidate layers with lowest BI scores
3. Use greedy algorithm to close entire layers and select MHA candidate layers
   - First, turn off FFN for all candidate layers
   - For each layer, evaluate loss with entire layer closed
   - Select layers with minimum loss as MHA candidates

### Stage 2: Summary of Results

- Display selected MHA removal layers
- Display selected FFN removal layers
- Calculate and display total removed layers
- Optionally evaluate perplexity on the pruned model

## Automatic Layer Calculation

When `mha_remove_layers` and `ffn_remove_layers` are not specified, the script automatically calculates them based on:

1. **Target prune rate**: The desired percentage of parameters to remove
2. **Parameter ratio**: The ratio between FFN and MHA parameters (can be auto-calculated from model config)
3. **Model configuration**: Hidden size, intermediate size, attention heads, etc.

The algorithm searches for the best combination of MHA and FFN layers to remove that:
- Achieves the target prune rate as closely as possible
- Maintains the specified FFN/MHA parameter ratio
- Ensures the total removed layers don't exceed the model's total layers

## Output

The script provides detailed output including:

- Model parameter information
- BI scores for all layers
- Selected candidate layers
- Pruning layer calculation results (if auto-calculated)
- Round-by-round progress during greedy selection
- Final pruning results (MHA and FFN removal layers)
- Perplexity evaluation results (if enabled)

## Requirements

- Python 3.8+
- PyTorch 2.5.1+
- CUDA (for GPU acceleration)
- See `requirements.txt` for full list of dependencies

## Example

```bash
# Automatic calculation with 25% prune rate
python ppllms.py \
    --model_name /workspace/models/meta-llama/Llama-2-13b-hf \
    --num_blocks 40 \
    --prune_rate 0.25

# Manual specification
python ppllms.py \
    --model_name /workspace/models/meta-llama/Llama-2-13b-hf \
    --num_blocks 40 \
    --mha_remove_layers 10 \
    --ffn_remove_layers 10 \
    --nsamples 256 \
    --eval_ppl True
```
