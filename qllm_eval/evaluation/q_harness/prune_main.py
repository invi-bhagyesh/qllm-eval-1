import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc
from qllm_eval.evaluation.q_harness.lm_eval_adaptor import LMEvalAdaptor
from qllm_eval.evaluation.q_harness.prune_utils import block_influence, get_model_layers, compute_layer_importance, remove_layers, prune_model    
from lm_eval import evaluator

parser = argparse.ArgumentParser()
# Existing quantization arguments
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--output_path", type=str, help="path to save the quantized model")
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--tasks", type=str, default="truthfulqa_mc")
parser.add_argument("--metrics", type=str, default="mc1,mc2")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)

# New pruning arguments
parser.add_argument("--enable_pruning", action="store_true", help="Enable layer pruning")
parser.add_argument("--layers_path", type=str, default="model.layers", 
                    help="Path to model layers in dot notation (e.g., 'model.layers')")
parser.add_argument("--n_prune_layers", type=int, default=0, 
                    help="Number of layers to prune")
parser.add_argument("--pruning_method", type=str, default="importance", 
                    choices=["importance", "angular", "reverse"], 
                    help="Pruning method: 'importance' (block influence), 'angular' (angular distance), or 'reverse' (remove last n layers)")
parser.add_argument("--calibration_dataset", type=str, default="wikitext",
                    help="Dataset for calibration (wikitext, c4, etc.) - not used for reverse pruning")
parser.add_argument("--n_calibration_samples", type=int, default=1000,
                    help="Number of calibration samples - not used for reverse pruning")
parser.add_argument("--pruning_stride", type=int, default=256,
                    help="Stride for sliding window during importance computation - not used for reverse pruning")
parser.add_argument("--pruning_max_seq_len", type=int, default=1024,
                    help="Maximum sequence length for pruning calibration - not used for reverse pruning")
parser.add_argument("--pruning_batch_size", type=int, default=1,
                    help="Batch size for pruning calibration - not used for reverse pruning")

args = parser.parse_args()


def main():
    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if 'falcon' in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64

    # Build model and tokenizer
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

    # Step 1: Quantize model FIRST
    print("\n" + "="*60)
    print("STEP 1: QUANTIZATION")
    print("="*60)
    model = quantize_model(model, args)
    print("Quantization complete!")

    # Step 2: Prune model (if enabled)
    removed_layers = []
    model_path_for_eval = args.model_path  # Default to original path
    
    if args.enable_pruning and args.n_prune_layers > 0:
        print("\n" + "="*60)
        print("STEP 2: PRUNING")
        print("="*60)
        print(f"Pruning method: {args.pruning_method}")
        if args.pruning_method == "reverse":
            print("NOTE: Reverse pruning doesn't require calibration data!")
        
        model, removed_layers = prune_model(model, args, enc)
        print("Pruning complete!")
        
        # **CRITICAL FIX: Save and reload the pruned model**
        print("\nSaving pruned model to temporary location for proper cache initialization...")
        temp_path = args.output_path if args.output_path else "./temp_pruned_model"
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        
        model.save_pretrained(temp_path, safe_serialization=False)
        enc.save_pretrained(temp_path)
        
        # Save pruning info
        pruning_info = {
            "removed_layers": removed_layers,
            "n_prune_layers": args.n_prune_layers,
            "pruning_method": args.pruning_method,
            "remaining_layers": len(get_model_layers(model, args.layers_path))
        }
        with open(os.path.join(temp_path, "pruning_info.json"), 'w') as f:
            json.dump(pruning_info, f, indent=2)
        
        print(f"Model saved to: {temp_path}")
        
        # Reload the model to ensure proper initialization
        print("Reloading pruned model for evaluation...")
        del model  # Free memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Reload with proper configuration
        model, enc = build_model_and_enc(temp_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
        model_path_for_eval = temp_path
        print("Pruned model reloaded successfully!")
        
    else:
        print("\nSkipping pruning (not enabled or n_prune_layers=0)")
        
        # Step 3: Save the model (if output path provided and no pruning)
        if args.output_path:
            print("\n" + "="*60)
            print("STEP 3: SAVING MODEL")
            print("="*60)
            print(f"Saving to: {args.output_path}")
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            model.save_pretrained(args.output_path, safe_serialization=False)
            enc.save_pretrained(args.output_path)
            print("Model saved!")

    # Step 4: Evaluation
    print("\n" + "="*60)
    print("STEP 4: EVALUATION")
    print("="*60)
    
    # Use the correct model path (original or temp pruned location)
    lm_eval_model = LMEvalAdaptor(model_path_for_eval, model, enc, 1)

    if args.tasks is not None:
        task_names = args.tasks.split(",")

        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=1,
            no_cache=True,
            num_fewshot=0,
        )
        
        # Save results
        for task_name in task_names:
            quant_str = "kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit)
            if removed_layers:
                output_dir = "{}/{}/pruned_{}".format(task_name, args.model_path, quant_str)
                output_file = "{}/pruned_{}_layers.jsonl".format(output_dir, len(removed_layers))
            else:
                output_dir = "{}/{}".format(task_name, args.model_path)
                output_file = "{}/{}.jsonl".format(output_dir, quant_str)
            
            print("* Output: ", output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            result_dict = results['results'][task_name].copy()
            if removed_layers:
                result_dict['pruning_info'] = {
                    'removed_layers': removed_layers,
                    'n_layers_removed': len(removed_layers),
                    'pruning_method': args.pruning_method
                }
            
            with open(output_file, 'w') as f:
                f.write(json.dumps(result_dict) + "\n")
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        if removed_layers:
            print(f"Pruned layers: {removed_layers}")
            print(f"Pruning method: {args.pruning_method}")
        print(f"Quantization: kv_{args.kv_bit}_w_{args.w_bit}_a_{args.a_bit}")
        print("\nTask Results:")
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
