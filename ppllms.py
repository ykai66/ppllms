import fire
import copy
import time
import os
from datetime import datetime
"""
ppllms: Two-stage pruning strategy
1. Stage 1: Use layer closure evaluation to select MHA candidate layers
2. Stage 2: Summary of results
"""
import torch
import torch.nn as nn
import pandas as pd

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace, block_replace_bi, final_replace, \
    final_turn_off_ffn, block_replace_final, turn_off_layer_final, turn_on_layer_final, turn_off_ffn_final, turn_on_ffn_final
from utils.data_utils import *
from utils.block_remove import block_remove
from utils.eval_utils import load_and_eval_ppl

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@torch.no_grad()
def get_loss(model, testenc, bs=1, device=None):
    """
    get_loss function copied from sleb.py
    """
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen
    # List to store negative log likelihoods
    losses = []

    # Loop through each batch
    for i in range(0, nsamples, bs):
        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        loss = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        losses.append(loss)

    # Compute sum of negative log_likelihood
    loss_sum = torch.stack(losses).sum()

    return loss_sum.item()


def calculate_all_bi_scores(model, dataloader, device):
    """
    Calculate BI scores for all layers
    """
    print(f"\n=== Calculating BI Scores for All Layers ===")
    
    # Calculate BI scores
    testenc = dataloader.input_ids
    inputs = testenc[:, :model.seqlen].to(device)
    inputs = inputs.reshape(1, model.seqlen)
    
    # Forward pass to calculate BI scores
    with torch.no_grad():  # Ensure no gradient computation to save memory
        lm_logits = model(inputs).logits
        block_importance_scores = [layer.importance for layer in model.model.layers]
    
    print(f"BI scores for all {len(block_importance_scores)} layers: {[f'{score:.6f}' for score in block_importance_scores]}")
    
    # Clear memory
    del lm_logits
    torch.cuda.empty_cache()
    
    return block_importance_scores


def select_lowest_bi_layers(bi_scores, num_layers):
    """
    Select specified number of layers with lowest BI scores
    """
    print(f"\n=== Selecting {num_layers} Layers with Lowest BI Scores ===")
    
    # Create list of (index, score) tuples
    bi_scores_with_indices = [(i, score) for i, score in enumerate(bi_scores)]
    
    # Sort by BI scores (ascending, lower scores are less important)
    sorted_bi_scores = sorted(bi_scores_with_indices, key=lambda x: x[1])
    
    # Select num_layers layers with lowest BI scores
    selected_layers = [idx for idx, score in sorted_bi_scores[:num_layers]]
    selected_scores = [score for idx, score in sorted_bi_scores[:num_layers]]
    
    print(f"Selected {num_layers} layers with lowest BI scores: {selected_layers}")
    print(f"BI scores of selected layers: {[f'{score:.6f}' for score in selected_scores]}")
    
    return selected_layers



def calculate_model_params(model):
    """
    Calculate the number of parameters in the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_layer_params(config):
    """
    Calculate the number of parameters for a single MHA and FFN layer based on model configuration
    
    Args:
        config: Model config object containing hidden_size, intermediate_size, num_attention_heads, etc.
    
    Returns:
        mha_params_per_layer: Number of parameters in a single MHA layer
        ffn_params_per_layer: Number of parameters in a single FFN layer
        layer_ratio: FFN parameters / MHA parameters
    """
    hidden_size = config.hidden_size
    intermediate_size = getattr(config, 'intermediate_size', None)
    
    # If intermediate_size is not available, use default value (usually hidden_size * 8/3, rounded up)
    if intermediate_size is None:
        intermediate_size = int(hidden_size * 8 / 3)
    
    # Calculate MHA parameters
    # MHA contains: Q, K, V, O four projection matrices, each is hidden_size x hidden_size
    # Note: For grouped attention (GQA), calculation may differ, here we assume standard attention
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    
    # For standard attention, each head dimension is hidden_size / num_attention_heads
    head_dim = hidden_size // num_attention_heads
    
    # Q projection: hidden_size x hidden_size
    # K, V projection: If using GQA, then hidden_size x (num_key_value_heads * head_dim)
    # O projection: hidden_size x hidden_size
    if num_key_value_heads == num_attention_heads:
        # Standard multi-head attention
        qkv_params = 3 * hidden_size * hidden_size  # Q, K, V
    else:
        # Grouped Query Attention (GQA)
        q_params = hidden_size * hidden_size  # Q
        kv_params = 2 * hidden_size * (num_key_value_heads * head_dim)  # K, V
        qkv_params = q_params + kv_params
    
    o_params = hidden_size * hidden_size  # O
    mha_params_per_layer = qkv_params + o_params
    
    # Calculate FFN parameters
    # FFN contains: gate_proj, up_proj, down_proj
    # gate_proj: hidden_size x intermediate_size
    # up_proj: hidden_size x intermediate_size
    # down_proj: intermediate_size x hidden_size
    ffn_params_per_layer = 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
    # Simplified calculation: ffn_params_per_layer = 3 * hidden_size * intermediate_size
    
    # Calculate ratio
    layer_ratio = ffn_params_per_layer / mha_params_per_layer if mha_params_per_layer > 0 else 0
    
    return mha_params_per_layer, ffn_params_per_layer, layer_ratio


def calculate_pruning_layers(model, prune_rate, mha_ffn_ratio, num_blocks):
    """
    Automatically calculate the number of MHA and FFN layers to remove based on target prune rate and MHA:FFN parameter ratio
    
    Args:
        model: Model object
        prune_rate: Target pruning rate (default 0.25, i.e., 25%)
        mha_ffn_ratio: Ratio of FFN parameters / MHA parameters (default 2.0, i.e., FFN is 2x MHA)
        num_blocks: Total number of layers
    
    Returns:
        mha_remove_layers: Number of MHA layers to remove
        ffn_remove_layers: Number of FFN layers to remove
        actual_prune_rate: Actual pruning rate achieved
    """
    print(f"\n=== Calculating Pruning Layers ===")
    print(f"Target prune rate: {prune_rate*100:.1f}%")
    
    # Calculate parameters per layer
    config = model.config
    mha_params_per_layer, ffn_params_per_layer, actual_ratio = calculate_layer_params(config)
    
    # Use actual ratio instead of input ratio
    use_ratio = actual_ratio if mha_ffn_ratio is None else mha_ffn_ratio
    
    print(f"FFN/MHA parameter ratio: {use_ratio:.3f} {'(auto-calculated)' if mha_ffn_ratio is None else '(user-specified)'}")
    
    print(f"\nModel configuration:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"\nParameter calculation:")
    print(f"  MHA params per layer: {mha_params_per_layer:,}")
    print(f"  FFN params per layer: {ffn_params_per_layer:,}")
    print(f"  Actual FFN/MHA ratio: {actual_ratio:.4f}")
    
    # Calculate total model parameters
    total_params, trainable_params = calculate_model_params(model)
    print(f"  Total model params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Calculate target parameters to remove
    target_remove_params = total_params * prune_rate
    
    # Iteratively search for best layer combination
    # Goal: mha_remove * mha_params + ffn_remove * ffn_params ≈ target_remove_params
    # Constraint: ffn_remove / mha_remove ≈ use_ratio (if mha_remove > 0)
    
    best_mha_layers = 0
    best_ffn_layers = 0
    best_diff = float('inf')
    
    # Search range: mha from 1 to num_blocks-1, ffn from 1 to num_blocks-1
    for mha_remove in range(1, num_blocks):
        # Calculate initial estimate for ffn_remove based on ratio
        if use_ratio > 0:
            ffn_remove_estimate = int(round(mha_remove * use_ratio))
        else:
            ffn_remove_estimate = mha_remove * 2
        
        # Search around the estimate
        search_range = max(1, int(ffn_remove_estimate * 0.3))
        for ffn_remove in range(max(1, ffn_remove_estimate - search_range), 
                                min(num_blocks, ffn_remove_estimate + search_range + 1)):
            if mha_remove + ffn_remove > num_blocks:
                continue
                
            # Calculate actual parameters removed
            removed_params = mha_remove * mha_params_per_layer + ffn_remove * ffn_params_per_layer
            actual_prune = removed_params / total_params
            
            # Calculate difference from target prune rate
            diff = abs(actual_prune - prune_rate)
            
            # Check if ratio requirement is met (allow some error)
            if mha_remove > 0:
                actual_layer_ratio = ffn_remove / mha_remove
                ratio_diff = abs(actual_layer_ratio - use_ratio) / use_ratio
                # Ratio error should not exceed 20%
                if ratio_diff > 0.2:
                    continue
            
            if diff < best_diff:
                best_diff = diff
                best_mha_layers = mha_remove
                best_ffn_layers = ffn_remove
    
    # Calculate actual pruning rate
    removed_params = best_mha_layers * mha_params_per_layer + best_ffn_layers * ffn_params_per_layer
    actual_prune_rate = removed_params / total_params
    
    print(f"\n=== Pruning Layer Calculation Results ===")
    print(f"MHA layers to remove: {best_mha_layers}")
    print(f"FFN layers to remove: {best_ffn_layers}")
    print(f"Total layers to process: {best_mha_layers + best_ffn_layers}")
    print(f"Target prune rate: {prune_rate*100:.2f}%")
    print(f"Actual prune rate: {actual_prune_rate*100:.2f}%")
    print(f"Difference: {abs(actual_prune_rate - prune_rate)*100:.2f}%")
    print(f"MHA params removed: {best_mha_layers * mha_params_per_layer:,}")
    print(f"FFN params removed: {best_ffn_layers * ffn_params_per_layer:,}")
    print(f"Total params removed: {removed_params:,}")
    if best_mha_layers > 0:
        print(f"Actual layer ratio (FFN/MHA): {best_ffn_layers / best_mha_layers:.3f}")
        print(f"Target ratio: {use_ratio:.3f}")
    
    return best_mha_layers, best_ffn_layers, actual_prune_rate

def progressive_pruning(model, removeffn, removemha):
    removal_list = removemha
    removal_ffn_list = removeffn

    model = block_replace(model)

    model.eval()
    model = block_remove(model, copy.deepcopy(removal_list))
    removal_set = set(removal_list)
    removal_ffn_set = set(removal_ffn_list)

    # FFN removed layers include all layers that need FFN removal, without excluding MHA parts
    removal_ffn_only_list = list(removal_ffn_set)

    new_removal_ffn_only_list = [idx - sum(1 for k in removal_list if k < idx) for idx in removal_ffn_only_list]

    model = final_replace(model)

    for i in new_removal_ffn_only_list:
        final_turn_off_ffn(model, i)


def greedy_layer_pruning_stage1(model, dataloader, device, candidate_layers, num_layers_to_remove):
    """
    Stage 1: Use greedy algorithm to close entire layers and select MHA candidate layers
    Logic:
    1. First, turn off FFN for all candidate_layers
    2. For each layer, turn_on_ffn + turn_off_layer, then calculate loss
    3. After calculation, turn_on_layer + turn_off_ffn to restore state
    """
    print(f"\n=== Stage 1: Greedy Layer Pruning (Select MHA Candidates) ===")
    print(f"Candidate layers: {candidate_layers}")
    print(f"Target: Select {num_layers_to_remove} layers as MHA candidates")
    
    # First, turn off FFN for all candidate_layers
    for layer_idx in candidate_layers:
        turn_off_ffn_final(model, layer_idx)
    
    alive_list = candidate_layers.copy()
    selected_layers = []
    
    all_losses = []
    testenc = dataloader.input_ids
    nsamples = testenc.numel() // model.seqlen
    
    for round_num in range(num_layers_to_remove):
        print(f"Round {round_num + 1}/{num_layers_to_remove}: Evaluating {len(alive_list)} layers...")
        min_loss = 1e99
        min_loss_idx = -1
        
        # Try to close each entire layer
        for j in range(len(alive_list)):
            layer_idx = alive_list[j]
            print(f"  Testing layer {layer_idx} ({j+1}/{len(alive_list)})...", end=' ', flush=True)
            
            # Current layer state: FFN is already turned off
            # 1. First turn on FFN (because we need to close the entire layer)
            turn_on_ffn_final(model, layer_idx)
            # 2. Close the entire layer
            turn_off_layer_final(model, layer_idx)
            
            # Calculate loss (at this point: this layer is entirely closed, other candidate layers' FFN are closed)
            loss = get_loss(model, dataloader, bs=1, device=device)
            torch.cuda.empty_cache()
            
            print(f"Loss={loss:.3f}")
            
            # Update minimum loss
            if loss < min_loss:
                min_loss = loss
                min_loss_idx = j
            
            # Restore state: turn on layer, turn off FFN
            turn_on_layer_final(model, layer_idx)
            turn_off_ffn_final(model, layer_idx)
        
        # Select layer with minimum loss
        selected_layer = alive_list[min_loss_idx]
        print(f"  Selected layer {selected_layer} with loss {min_loss:.3f}")
        
        # Permanently close selected layer (turn_on_ffn + turn_off_layer)
        turn_on_ffn_final(model, selected_layer)
        turn_off_layer_final(model, selected_layer)
        
        selected_layers.append(selected_layer)
        alive_list.remove(selected_layer)
        
        round_data = {
            'Round': round_num + 1,
            'Selected_Layer': selected_layer,
            'Loss': float(min_loss)
        }
        all_losses.append(round_data)
    
    return selected_layers, alive_list



def ppllms(
        model_name: str = '/workspace/models/meta-llama/Llama-2-7b-hf',
        num_blocks: int = 32,
        prune_rate: float = 0.25,
        seed: int = 0,
        nsamples: int = 128,
        dataset: str = '/workspace/datasets/allenai/c4',
        eval_ppl: bool = True,
        mha_remove_layers: int = None,  
        ffn_remove_layers: int = None,  
        mha_ffn_ratio: float = 2.0  
):

    print(f"=== ppllms: Two-Stage Pruning Strategy with Auto-Calculation ===")
    print(f"Stage 1: Close entire layer to select MHA candidates")
    print(f"Stage 2: Greedily select MHA to remove (with FFN closed)")
    
    # Load model (only for parameter calculation, will reload later)
    print(f"\n=== Loading Model for Parameter Calculation ===")
    model = get_llm(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model is already loaded with device map
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        print("Model is already loaded with device map, skipping .to(device)")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
    else:
        model.to(device)
    
    # calculate pruning layers automatically
    if mha_remove_layers is None or ffn_remove_layers is None:
        mha_remove_layers, ffn_remove_layers, actual_prune_rate = calculate_pruning_layers(
            model, prune_rate=prune_rate, mha_ffn_ratio=mha_ffn_ratio, num_blocks=num_blocks
        )
        print(f"\nAuto-calculated pruning configuration:")
        print(f"  MHA layers to remove: {mha_remove_layers}")
        print(f"  FFN layers to remove: {ffn_remove_layers}")
        print(f"  Actual prune rate: {actual_prune_rate*100:.2f}%")
    else:
        print(f"\nUsing manually specified pruning configuration:")
        print(f"  MHA layers to remove: {mha_remove_layers}")
        print(f"  FFN layers to remove: {ffn_remove_layers}")
    
    # release model memory
    del model
    torch.cuda.empty_cache()
    
    # calculate number of layers to remove
    num_remove_blocks = ffn_remove_layers
    print(f"\nPrune rate: {prune_rate}")
    print(f"Total blocks: {num_blocks}")
    print(f"Target removal: {num_remove_blocks} total")
    # Load model
    model = get_llm(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model is already loaded with device map
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
    else:
        model.to(device)
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"Loaded Model: {model.name}")

    # Replace with BI model
    model = block_replace_bi(model)
    model.eval()

    # Load data
    dataloader = get_trainloaders(dataset,
                                  nsamples=nsamples,
                                  seed=seed,
                                  model=model_name,
                                  )

    # Stage 1: Calculate BI scores and select candidate layers
    all_bi_scores = calculate_all_bi_scores(model, dataloader, device)
    candidate_layers = select_lowest_bi_layers(all_bi_scores, num_remove_blocks)
    
    # Stage 1: Use greedy algorithm to close entire layers and select MHA candidate layers
    print(f"\n=== Stage 1: Greedy Layer Pruning (Select MHA Candidates) ===")
    
    # Use block_replace_final (on_off_llama_final) to independently control layer, ffn, mha
    model = block_replace_final(model)
    
    # Only move model if it's not already loaded with device map
    if not (hasattr(model, 'hf_device_map') and model.hf_device_map):
        model.to(device)
    model.eval()
    
    mha_candidate_layers, alive_list = greedy_layer_pruning_stage1(model, dataloader, device, candidate_layers, mha_remove_layers)
    
    # Note: In stage1, we have already:
    # 1. Turned off FFN for all candidate_layers
    # 2. Selected mha_candidate_layers have been turn_off_layer (entire layer closed)
    # 3. Remaining layers' FFN remain closed
    
    print(f"\n=== Stage 2: Summary of Results ===")
    mha_removal_layers = mha_candidate_layers  # Selected layers remove MHA
    ffn_removal_layers = alive_list  # Remaining layers remove FFN
    print(f"MHA removed layers: {mha_removal_layers}")
    print(f"FFN removed layers: {ffn_removal_layers}")
    print(f"Total removed: {len(mha_removal_layers)} MHA + {len(ffn_removal_layers)} FFN = {len(mha_removal_layers) + len(ffn_removal_layers)} total")

    if eval_ppl:
        removeffn = ffn_removal_layers
        removemha = mha_removal_layers

        progressive_pruning(model, removeffn, removemha)
        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
        print(f"C4 PPL = {c4_ppl:.2f}")

if __name__ == "__main__":
    fire.Fire(ppllms)

