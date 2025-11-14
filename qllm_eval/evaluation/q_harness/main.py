import os
import json
import argparse
import datasets
import sys
import torch
import pandas as pd

from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc
from qllm_eval.evaluation.q_harness.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator


parser = argparse.ArgumentParser()
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
parser.add_argument("--dataset", type=str, default=None, help="dataset name for DPO like 'anthropic'")
parser.add_argument("--dpo_path", type=str, default=None,
                    help="path to an existing DPO fine-tuned model; if provided, DPO training will be skipped")
args = parser.parse_args()


def create_dtype_patch(model):
    """Create a patched forward function for a model to ensure correct dtypes"""
    original_forward = model.forward
    
    def patched_forward(input_ids=None, **kwargs):
        # Ensure input_ids is long type
        if input_ids is not None:
            if not input_ids.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                input_ids = input_ids.long()
        
        # Ensure attention_mask is also correct type if present
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            if not kwargs['attention_mask'].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                kwargs['attention_mask'] = kwargs['attention_mask'].long()
        
        # Ensure labels is also correct type if present
        if 'labels' in kwargs and kwargs['labels'] is not None:
            if not kwargs['labels'].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                kwargs['labels'] = kwargs['labels'].long()
        
        return original_forward(input_ids=input_ids, **kwargs)
    
    return original_forward, patched_forward


def main():
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if 'falcon' in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64

    ###########################################################################
    if args.dataset == "anthropic":
        if args.dpo_path is not None:
            print(f"Loading existing DPO model from {args.dpo_path}, skipping DPO training...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = build_model_and_enc(args.dpo_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            from datasets import load_dataset, Dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from trl import DPOTrainer, DPOConfig

            dpo_output_dir = "./dpo_finetuned_anthropic"

            print("Loading Anthropic HH-RLHF dataset for DPO...")
            ds = load_dataset("Anthropic/hh-rlhf", split="train[:10%]")

            # Load tokenizer and model first
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 to avoid dtype issues
            )

            # Prepare data in the format DPO expects (keep original text fields)
            pairs = []
            for item in ds:
                pairs.append({
                    "prompt": item.get("input", ""),
                    "chosen": item["chosen"],
                    "rejected": item["rejected"]
                })

            train_data = Dataset.from_pandas(pd.DataFrame(pairs))

            print("Starting DPO fine-tuning using TRL...")

            # Patch the main model's forward
            original_model_forward, patched_model_forward = create_dtype_patch(model)
            model.forward = patched_model_forward
            
            # Also patch torch.gather to ensure index is always long
            import torch.nn.functional as F
            original_gather = torch.gather
            
            def patched_gather(input, dim, index, **kwargs):
                # Ensure index is long type for gather operations
                if index is not None and not index.dtype in [torch.long, torch.int64]:
                    index = index.long()
                return original_gather(input, dim, index, **kwargs)
            
            torch.gather = patched_gather
            
            # Configure DPO training
            training_args = DPOConfig(
                output_dir=dpo_output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                remove_unused_columns=False,
                logging_steps=10,
                gradient_accumulation_steps=1,
                learning_rate=5e-6,
                warmup_steps=2,
                fp16=False,
                bf16=False, 
                save_strategy="epoch",     # save only after the last epoch
                save_total_limit=1,        # keep only the last checkpoint
                save_safetensors=True,
                save_optimizer=False,
                save_on_each_node=False,
                eval_strategy="no",
                report_to='none',
                max_length=512,
                max_prompt_length=256,
                precompute_ref_log_probs=False,
                dataset_num_proc=1,
            )

            # Load a frozen reference model from the *base* checkpoint
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            ref_model.requires_grad_(False)
            
            # Patch the reference model's forward as well
            original_ref_forward, patched_ref_forward = create_dtype_patch(ref_model)
            ref_model.forward = patched_ref_forward

            # Create DPO trainer
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=None,
            )

            # Train the model
            trainer.train()
            
            # Restore original forwards and gather before saving
            model.forward = original_model_forward
            ref_model.forward = original_ref_forward
            torch.gather = original_gather
            
            # Save the fine-tuned model
            model.save_pretrained(dpo_output_dir)
            tokenizer.save_pretrained(dpo_output_dir)
            print(" DPO fine-tuning complete on Anthropic dataset. Proceeding to quantization...")

            # Load the fine-tuned model for quantization
            model, tokenizer = build_model_and_enc(dpo_output_dir, args.use_flash_attn, args.kv_bit, args.kv_group_size)

    elif args.dataset == "pku":
        if args.dpo_path is not None:
            print(f"Loading existing DPO model from {args.dpo_path}, skipping DPO training...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = build_model_and_enc(args.dpo_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Loading PKU dataset for DPO...")
            from datasets import load_dataset
            import pandas as pd
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from trl import DPOTrainer, DPOConfig
            from datasets import Dataset

            dpo_output_dir = f"./dpo_finetuned_{args.tasks}"

            # Example: PKU dataset structure
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")["train"]
            ds = ds.shuffle(seed=42).select(range(int(len(ds) * 0.25)))

            pairs = []

            for item in ds:
                chosen = item["response_0"] if item["better_response_id"] == 0 else item["response_1"]
                rejected = item["response_1"] if item["better_response_id"] == 0 else item["response_0"]

                pairs.append({
                    "prompt": item["prompt"],
                    "chosen": chosen,
                    "rejected": rejected
                })

            pairs_df = pd.DataFrame(pairs)
            print(pairs_df.head())

            # Load base model + tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(args.model_path)

            # Prepare train data
            train_data = [{
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"]
            } for _, row in pairs_df.iterrows()]

            print("Starting DPO fine-tuning using TRLX...")

            training_args = DPOConfig(
                output_dir=dpo_output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=2,     # slightly larger for speed
                per_device_eval_batch_size=2,
                remove_unused_columns=False,
                logging_steps=20,
                gradient_accumulation_steps=2,     # mild accumulation
                learning_rate=7e-6,                # tiny bump for faster convergence
                warmup_steps=5,
                fp16=True,                         # still saves memory
                save_steps=1000,
                eval_strategy="no",
                report_to='none',
                gradient_checkpointing=False,      # disable checkpointing for speed
            )


            tokenizer.pad_token = tokenizer.eos_token
            train_data = Dataset.from_list(train_data)

            def preprocess(example):
                return {
                    "prompt_ids": tokenizer(
                        example["prompt"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    )["input_ids"],
                    "chosen_ids": tokenizer(
                        example["chosen"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    )["input_ids"],
                    "rejected_ids": tokenizer(
                        example["rejected"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    )["input_ids"],
                }

            train_data = train_data.map(preprocess, batched=False)

            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=None
            )

            trainer.train()
            model.save_pretrained(dpo_output_dir)
            tokenizer.save_pretrained(dpo_output_dir)
            print("DPO fine-tuning complete. Proceeding to quantization...")


    ###########################################################################
    else:
        model, tokenizer = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

    ###########################################################################
    # quantize model
    model = quantize_model(model, args)

    # evaluation
    lm_eval_model = LMEvalAdaptor(args.model_path, model, tokenizer, 1)

    if args.tasks is not None:
        task_names = args.tasks.split(",")

        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=1,
            cache_requests=None,
            num_fewshot=0,
        )

        for task_name in task_names:
            output_path = f"{task_name}/{args.model_path}/kv_{args.kv_bit}_w_{args.w_bit}_a_{args.a_bit}.jsonl"
            print("* Output:", output_path)
            os.makedirs(f"{task_name}/{args.model_path}", exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json.dumps(results['results'][task_name]) + "\n")


if __name__ == "__main__":
    main()