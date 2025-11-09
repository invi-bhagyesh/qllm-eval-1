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
            ds = load_dataset("Anthropic/hh-rlhf", split="train[:2%]")

            pairs = []
            for item in ds:
                pairs.append({
                    "prompt": item["input"] if "input" in item else "",
                    "chosen": item["chosen"],
                    "rejected": item["rejected"]
                })

            train_data = Dataset.from_pandas(pd.DataFrame(pairs))

            print("Starting DPO fine-tuning using TRL...")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

            # Tokenization function for DPO
            def encode_texts(example):
                # Tokenize prompt, chosen, and rejected
                prompt_tokenized = tokenizer(
                    example["prompt"], 
                    truncation=True, 
                    max_length=512
                )
                chosen_tokenized = tokenizer(
                    example["chosen"], 
                    truncation=True, 
                    max_length=512
                )
                rejected_tokenized = tokenizer(
                    example["rejected"], 
                    truncation=True, 
                    max_length=512
                )
                
                # DPOTrainer expects these exact field names
                return {
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                    "prompt_input_ids": prompt_tokenized["input_ids"],
                    "prompt_attention_mask": prompt_tokenized["attention_mask"],
                    "chosen_input_ids": chosen_tokenized["input_ids"],
                    "chosen_attention_mask": chosen_tokenized["attention_mask"],
                    "rejected_input_ids": rejected_tokenized["input_ids"],
                    "rejected_attention_mask": rejected_tokenized["attention_mask"],
                }

            # Apply the encoding
            train_data = train_data.map(encode_texts, remove_columns=train_data.column_names)

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
                save_steps=500,
                eval_strategy="no",
                report_to='none',
                max_length=512,
                max_prompt_length=512,
            )

            # Create DPO trainer
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=None
            )

            # Train the model
            trainer.train()
            
            # Save the fine-tuned model
            model.save_pretrained(dpo_output_dir)
            tokenizer.save_pretrained(dpo_output_dir)
            print("✅ DPO fine-tuning complete on Anthropic dataset. Proceeding to quantization...")

            # Load the fine-tuned model for quantization
            model, tokenizer = build_model_and_enc(dpo_output_dir, args.use_flash_attn, args.kv_bit, args.kv_group_size)

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