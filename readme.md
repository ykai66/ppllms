# PPllms

PPLlms is a two-stage pruning strategy for large language models. Code for paper: "PP-LLMs: A Progressive Pruning Approach with Medium-Granularity for Large Language Models"

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python ppllms.py \
    --model_name /path/to/model \
    --num_blocks 32 \
    --prune_rate 0.25
```

### Main Parameters

- `model_name`: Path to the model (default: `/workspace/models/meta-llama/Llama-2-7b-hf`)
- `num_blocks`: Number of blocks in the model (default: `32`)
- `prune_rate`: Target pruning rate (default: `0.25`)
- `mha_remove_layers`: Number of MHA layers to remove (optional, auto-calculated if not specified)
- `ffn_remove_layers`: Number of FFN layers to remove (optional, auto-calculated if not specified)
- `nsamples`: Number of samples for evaluation (default: `128`)
- `dataset`: Path to the dataset (default: `/workspace/datasets/allenai/c4`)
- `eval_ppl`: Whether to evaluate perplexity (default: `True`)

## Two-Stage Pruning Strategy

**Stage 1: Candidate layer Screening**
- Calculate Block Importance (BI) scores for all layers
- Select candidate layers with lowest BI scores
- Use greedy algorithm to select MHA candidate layers by closing entire layers

**Stage 2: pruning MHA layers**
- Display selected MHA and FFN removal layers
- Optionally evaluate perplexity on the pruned model

If `mha_remove_layers` and `ffn_remove_layers` are not specified, the script automatically calculates them based on the target prune rate and parameter ratio.

## Requirements

- Python 3.8+
- PyTorch 2.5.1+
- CUDA (for GPU acceleration)
