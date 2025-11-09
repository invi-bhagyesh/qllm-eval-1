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
            import torch.nn.functional as F

            dpo_output_dir = "./dpo_finetuned_anthropic"

            print("Loading Anthropic HH-RLHF dataset for DPO...")
            ds = load_dataset("Anthropic/hh-rlhf", split="train[:2%]")

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

            # Comprehensive patch for DPOTrainer to fix dtype issue
            from trl.trainer.dpo_trainer import DPOTrainer as OriginalDPOTrainer
            
            class FixedDPOTrainer(OriginalDPOTrainer):
                def concatenated_forward(self, model, batch):
                    """Override to ensure input_ids are Long tensors"""
                    # Get the concatenated inputs
                    concatenated_batch = self.concatenated_inputs(
                        batch,
                        is_encoder_decoder=self.is_encoder_decoder,
                        is_vision_model=self.is_vision_model,
                        label_pad_token_id=self.label_pad_token_id,
                        padding_value=self.padding_value,
                        device=self.accelerator.device,
                    )
                    
                    # Extract input_ids and ensure it's long type
                    input_ids = concatenated_batch["concatenated_input_ids"].long()
                    attention_mask = concatenated_batch.get("concatenated_attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.long()
                    
                    # Prepare model kwargs
                    model_kwargs = {}
                    if attention_mask is not None:
                        model_kwargs["attention_mask"] = attention_mask
                    
                    if self.is_encoder_decoder:
                        decoder_input_ids = concatenated_batch.get("concatenated_decoder_input_ids")
                        if decoder_input_ids is not None:
                            model_kwargs["decoder_input_ids"] = decoder_input_ids.long()
                        decoder_attention_mask = concatenated_batch.get("concatenated_decoder_attention_mask")
                        if decoder_attention_mask is not None:
                            model_kwargs["decoder_attention_mask"] = decoder_attention_mask.long()
                    
                    if self.is_vision_model:
                        pixel_values = concatenated_batch.get("concatenated_pixel_values")
                        pixel_attention_mask = concatenated_batch.get("concatenated_pixel_attention_mask")
                        if pixel_values is not None:
                            model_kwargs["pixel_values"] = pixel_values
                        if pixel_attention_mask is not None:
                            model_kwargs["pixel_attention_mask"] = pixel_attention_mask
                    
                    if self.aux_loss_enabled:
                        model_kwargs["output_router_logits"] = True
                    
                    # Call model with corrected dtypes
                    outputs = model(
                        input_ids,
                        **model_kwargs,
                        use_cache=False,
                    )
                    
                    return outputs
            
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
                save_steps=500,
                eval_strategy="no",
                report_to='none',
                max_length=512,
                max_prompt_length=256,
                precompute_ref_log_probs=False,
                dataset_num_proc=1,
            )

            # Create DPO trainer with the patched version
            trainer = FixedDPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=None,
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