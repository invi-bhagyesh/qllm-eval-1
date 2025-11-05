import os
import json
import argparse
import datasets  # Add this import
import sys

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
parser.add_argument("--dataset", type=str, default=None, help="dataset name for DPO like 'crows-pairs'")
args = parser.parse_args()



def main():
    # Set trust_remote_code globally for datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True  # Add this line
    
    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if 'falcon' in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64


    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

###########################################################################
    if args.dataset == "crows-pairs":
        from datasets import load_dataset
        import pandas as pd
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trlx.data import default_configs



        print("Loading crows-pairs dataset for DPO...")
        ds = load_dataset("BigScienceBiasEval/crows_pairs_multilingual")["test"]
        pairs = []

        for item in ds:
            chosen = item["sent_less"]
            rejected = item["sent_more"]
            if item["stereo_antistereo"] == "anti-stereotype":
                chosen, rejected = rejected, chosen

            pairs.append({
                "prompt": f"Bias type: {item['bias_type']}",  # minimal context
                "chosen": chosen,
                "rejected": rejected,
                "bias_type": item["bias_type"]
            })

        pairs_df = pd.DataFrame(pairs)
        print(pairs_df.head())

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(args.model_path)

        # Convert dataset to dict
        train_data = [{
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"]
        } for _, row in pairs_df.iterrows()]

        print("Starting DPO fine-tuning using TRLX...")

        # Define DPO config
        # Copy the default DPO config
        dpo_config = default_configs.dpo

        # Optionally override some values
        dpo_config.learning_rate = 5e-6
        dpo_config.batch_size = 2
        dpo_config.gradient_accumulation_steps = 4
        dpo_config.max_steps = 200
        dpo_config.logging_dir = "./trlx_logs"


        # Start DPO training
        trainer = trlx.train(
            model,
            train_data,
            reward_fn=None,  # optional custom reward function
            tokenizer=tokenizer,
            config=dpo_config
        )

        model.save_pretrained("./dpo_finetuned")
        print("DPO fine-tuning complete. Proceeding to quantization...")

###########################################################################
    # quantize model
    model = quantize_model(model, args)


    # # save the quantized model
    # if args.output_path:
    #     model.save_pretrained(args.output_path, safe_serialization=False)
    #     enc.save_pretrained(args.output_path)


    # evaluation
    lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, 1)


    if args.tasks is not None:
        task_names = args.tasks.split(",")


        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=1,
            cache_requests=None,
            num_fewshot=0,
        )
        # print(results)
        # print(evaluator.make_table(results))
        for task_name in task_names:
            output_path = "{}/{}/kv_{}_w_{}_a_{}.jsonl".format(task_name, args.model_path, args.kv_bit, args.w_bit, args.a_bit)
            print("* Output: ", output_path)
            if not os.path.exists("{}/{}".format(task_name, args.model_path)):
                os.makedirs("{}/{}".format(task_name, args.model_path))
            with open(output_path, 'w') as f:
                f.write(json.dumps(results['results'][task_name]) + "\n")



if __name__ == "__main__":
    main()
