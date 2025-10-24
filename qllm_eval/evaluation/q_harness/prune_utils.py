import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


def block_influence(in_hidden, out_hidden, angular=False):
    """
    Compute block influence metric between input and output hidden states.
    
    Args:
        in_hidden: Input hidden states
        out_hidden: Output hidden states
        angular: Whether to use angular distance
    
    Returns:
        Block influence score
    """
    if angular:
        # Compute cosine similarity for angular distance
        in_norm = in_hidden / (in_hidden.norm(dim=-1, keepdim=True) + 1e-8)
        out_norm = out_hidden / (out_hidden.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = (in_norm * out_norm).sum(dim=-1)
        # Convert similarity to distance (1 - similarity)
        return 1 - similarity
    else:
        # Use L2 distance
        return torch.norm(out_hidden - in_hidden, p=2, dim=-1)


def get_model_layers(model, layers_path):
    """Extract layers from model using dot notation path."""
    modules = layers_path.split(".")
    mod = model
    for m in modules:
        mod = getattr(mod, m)
    return mod


def compute_layer_importance(model, layers, dataloader, args, tokenizer):
    """
    Compute layer-wise importance scores using calibration data.
    
    Args:
        model: The language model
        layers: Model layers to evaluate
        dataloader: Calibration data loader
        args: Arguments containing pruning parameters
        tokenizer: Model tokenizer
    
    Returns:
        List of importance scores for each layer
    """
    importances = [0.0 for _ in layers]
    angular = (args.pruning_method == "angular")
    n = args.n_prune_layers if angular else 1
    
    print(f"Computing {'angular' if angular else 'standard'} importance scores...")
    
    for batch in tqdm(dataloader, desc="Processing calibration data"):
        prompts = batch['text']
        
        # Tokenize
        prompt_tokens = tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask
        
        max_prompt_len = max(len(t) for t in input_ids)
        
        # Sliding window approach
        for start in range(0, max_prompt_len, args.pruning_stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids
            inputs = input_ids[seq_ids, start:start+args.pruning_max_seq_len]
            attn = attn_mask[seq_ids, start:start+args.pruning_max_seq_len]
            
            # Get hidden states
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs.to(model.device),
                    attention_mask=attn.to(model.device),
                    output_hidden_states=True,
                )
            
            hiddens = outputs.hidden_states
            
            # Compute block influence
            for i in range(len(hiddens) - n):
                in_hidden = hiddens[i]
                out_hidden = hiddens[i+n]
                
                if angular:
                    # Use only last token for angular distance
                    in_hidden = in_hidden[:, -1:]
                    out_hidden = out_hidden[:, -1:]
                
                importances[i] += block_influence(
                    in_hidden,
                    out_hidden,
                    angular=angular
                ).sum().cpu().item()
    
    return importances


def remove_layers(layers, importances, n_prune, pruning_method="importance"):
    """
    Remove layers based on importance scores or pruning method and re-index remaining layers.
    
    Args:
        layers: Model layers (ModuleList)
        importances: Layer importance scores (can be None for reverse pruning)
        n_prune: Number of layers to prune
        pruning_method: Pruning method ('importance', 'angular', or 'reverse')
    
    Returns:
        List of removed layer indices
    """
    total_layers = len(layers)
    
    if pruning_method == "reverse":
        # Reverse-order pruning: remove the last n layers
        # Research shows this is surprisingly effective for LLMs
        layers_to_remove = list(range(total_layers - n_prune, total_layers))
        print(f"Reverse pruning: removing last {n_prune} layers (indices {layers_to_remove})")
        
    elif pruning_method == "angular":
        # For angular: find consecutive block with lowest importance
        assert importances is not None, "Importances required for angular pruning"
        start_layer = np.argsort(np.array(importances[:-n_prune+1]))[0]
        layers_to_remove = list(range(start_layer, start_layer + n_prune))
        print(f"Angular pruning: removing consecutive layers {layers_to_remove}")
        
    else:  # importance
        # For standard: remove layers with lowest importance scores
        assert importances is not None, "Importances required for importance-based pruning"
        layers_to_remove = np.argsort(np.array(importances))[:n_prune].tolist()
        print(f"Importance pruning: removing layers with lowest scores {layers_to_remove}")
    
    # Remove layers in reverse to avoid indexing errors
    for layer_idx in sorted(layers_to_remove, reverse=True):
        del layers[layer_idx]
    
    # **FIX: Re-index the remaining layers**
    for new_idx, layer in enumerate(layers):
        if hasattr(layer, 'layer_idx'):
            layer.layer_idx = new_idx
    
    return layers_to_remove


def fix_model_config(model, n_remaining_layers):
    """
    Update model configuration to reflect the new number of layers.
    
    Args:
        model: The pruned model
        n_remaining_layers: Number of layers after pruning
    """
    # Update config
    if hasattr(model, 'config'):
        model.config.num_hidden_layers = n_remaining_layers
    
    # For models with nested structure (e.g., LlamaForCausalLM)
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        model.model.config.num_hidden_layers = n_remaining_layers
    
    return model


def prune_model(model, args, tokenizer):
    """
    Prune model layers based on importance scores or pruning method.
    
    Args:
        model: The language model to prune
        args: Arguments containing pruning parameters
        tokenizer: Model tokenizer
    
    Returns:
        Tuple of (pruned_model, removed_layers)
    """
    # Get model layers
    layers = get_model_layers(model, args.layers_path)
    original_n_layers = len(layers)
    print(f"Total layers before pruning: {original_n_layers}")
    
    # For reverse pruning, we don't need to compute importances
    if args.pruning_method == "reverse":
        print(f"\nUsing reverse-order pruning (removing last {args.n_prune_layers} layers)...")
        print("Research shows this simple method is often very effective!")
        importances = None
        removed_layers = remove_layers(layers, importances, args.n_prune_layers, args.pruning_method)
    else:
        # Load calibration dataset for importance-based methods
        print(f"\nLoading calibration dataset: {args.calibration_dataset}...")
        if args.calibration_dataset == "wikitext":
            data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
            data = data.filter(lambda x: len(x['text'].strip()) > 100)
        elif args.calibration_dataset == "c4":
            data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            data = data.take(args.n_calibration_samples)
        else:
            raise ValueError(f"Unsupported calibration dataset: {args.calibration_dataset}")
        
        data = data.select(range(min(args.n_calibration_samples, len(data))))
        
        dataloader = DataLoader(
            data,
            batch_size=args.pruning_batch_size,
            shuffle=True,
        )
        
        # Compute importance scores
        importances = compute_layer_importance(model, layers, dataloader, args, tokenizer)
        
        # Remove layers based on importance
        print(f"\nRemoving {args.n_prune_layers} layers based on {args.pruning_method} scores...")
        removed_layers = remove_layers(layers, importances, args.n_prune_layers, args.pruning_method)
    
    # **FIX: Update model configuration**
    n_remaining_layers = len(layers)
    model = fix_model_config(model, n_remaining_layers)
    
    print(f"Removed layers: {removed_layers}")
    print(f"Remaining layers: {n_remaining_layers}")
    print(f"Model config updated: num_hidden_layers = {n_remaining_layers}")
    
    return model, removed_layers
