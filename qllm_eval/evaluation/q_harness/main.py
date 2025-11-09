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
parser.add_argument("--dpo_path", type=str, default=None,
                    help="path to an existing DPO fine-tuned model; if provided, DPO training will be skipped")
args = parser.parse_args()



def main():
    # Set trust_remote_code globally for datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True  # Add this line
    
    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if 'falcon' in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64


    # a hack here to auto set model group
    # model, tokenizer = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

###########################################################################
    if args.dataset == "crows-pairs":
        if args.dpo_path is not None:
            print(f"Loading existing DPO model from {args.dpo_path}, skipping DPO training...")
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = build_model_and_enc(args.dpo_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token   
        else:
            model, tokenizer = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

            dpo_output_dir = f"./dpo_finetuned_{args.tasks}"
            from datasets import load_dataset
            import pandas as pd
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print("Loading crows-pairs dataset for DPO...")
            ds = load_dataset("BigScienceBiasEval/crows_pairs_multilingual")["test"]
            pairs = []

            for item in ds:
                chosen = item["sent_less"]
                rejected = item["sent_more"]
                if item["stereo_antistereo"] == "anti-stereotype":
                    chosen, rejected = rejected, chosen

                pairs.append({
                    "prompt": f"Bias type: {item['bias_type']}",
                    "chosen": chosen,
                    "rejected": rejected,
                    "bias_type": item["bias_type"]
                })

            pairs_df = pd.DataFrame(pairs)
            print(pairs_df.head())

            # Load tokenizer and base model
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(args.model_path)

            # Prepare data
            train_data = [{
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"]
            } for _, row in pairs_df.iterrows()]

            print("Starting DPO fine-tuning using TRLX...")

            from trl import DPOTrainer, DPOConfig
            from datasets import Dataset

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
                report_to='none'
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
    elif args.dataset == "bbq":
        if args.dpo_path is not None:
            print(f"Loading existing DPO model from {args.dpo_path}, skipping DPO training...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = build_model_and_enc(args.dpo_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            model, tokenizer = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

            dpo_output_dir = f"./dpo_finetuned_{args.tasks}"
            from datasets import load_dataset
            import pandas as pd
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print("Loading BBQ dataset for DPO...")
            ds = load_dataset("bbq", "all")["train"]  # or 'validation', depending on availability
            pairs = []

            # Each example_id groups several related examples
            for example_id, group in pd.DataFrame(ds).groupby("example_id"):
                ambig = group[group["context_condition"] == "ambig"]

                for _, amb in ambig.iterrows():
                    prompt = f"{amb['context']}\n{amb['question']}"
                    ans_dict = amb["answer_info"]
                    # valid answers
                    answers = [
                        ans_dict["ans0"][0],
                        ans_dict["ans1"][0],
                        ans_dict["ans2"][0]
                    ]

                    chosen = answers[amb["label"]]
                    # unbiased choice is usually "unknown" or "Cannot be determined"
                    # reject any gendered/group answers if label is the unbiased one
                    for i, a in enumerate(answers):
                        if i == amb["label"]:
                            continue
                        if any(x.lower() in a.lower() for x in ["man", "woman", "male", "female"]):
                            pairs.append({
                                "prompt": prompt,
                                "chosen": chosen,
                                "rejected": a,
                                "category": amb["category"],
                                "context_condition": amb["context_condition"],
                                "question_polarity": amb["question_polarity"]
                            })

            pairs_df = pd.DataFrame(pairs)
            print(pairs_df.head())

            # Load tokenizer and base model
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(args.model_path)

            train_data = [{
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"]
            } for _, row in pairs_df.iterrows()]

            print("Starting DPO fine-tuning using TRLX...")

            from trl import DPOTrainer, DPOConfig
            from datasets import Dataset

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
                report_to='none'
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
    elif args.dataset == "winogender":
        if args.dpo_path is not None:
            print(f"Loading existing DPO model from {args.dpo_path}, skipping DPO training...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model, tokenizer = build_model_and_enc(args.dpo_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            model, tokenizer = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

            dpo_output_dir = f"./dpo_finetuned_{args.tasks}"
            from datasets import load_dataset
            import pandas as pd
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print("Loading Winogender dataset for DPO...")
            # You can replace this with your local CSV if needed:
            ds = load_dataset("winogender", split="train")  # or your own processed HuggingFace dataset
            df = pd.DataFrame(ds)

            pairs = []

            # Each sentence group has male, female, and neutral variants
            df["group_id"] = df["sentid"].apply(lambda x: ".".join(x.split(".")[:-1]))

            for group_id, group in df.groupby("group_id"):
                neutral = group[group["gender"] == "neutral"]
                male = group[group["gender"] == "male"]
                female = group[group["gender"] == "female"]

                if len(neutral) == 0:
                    continue

                neutral_row = neutral.iloc[0]
                prompt = neutral_row["sentence"]
                correct_target = neutral_row["target"]
                chosen = f"The pronoun '{neutral_row['pronoun']}' refers to the {correct_target}."

                # Add pairs with both gendered versions as rejected
                for _, row in pd.concat([male, female]).iterrows():
                    rejected = f"The pronoun '{row['pronoun']}' refers to the {row['target']}."
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "occupation": row["occupation"],
                        "participant": row["participant"]
                    })

            pairs_df = pd.DataFrame(pairs)
            print(pairs_df.head())

            # Load tokenizer and base model
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(args.model_path)

            # Prepare DPO training data
            train_data = [{
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"]
            } for _, row in pairs_df.iterrows()]

            print("Starting DPO fine-tuning using TRLX...")

            from trl import DPOTrainer, DPOConfig
            from datasets import Dataset

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
                report_to='none'
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

            if args.dpo_path is not None:
                print(f"Loading existing DPO model from {args.dpo_path}, skipping DPO training...")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                model, tokenizer = build_model_and_enc(args.dpo_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            else:
                model, tokenizer = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

                dpo_output_dir = f"./dpo_finetuned_{args.tasks}"
                import pandas as pd
                from transformers import AutoTokenizer, AutoModelForCausalLM

                print("Loading Winogender dataset for DPO...")
                df = pd.read_csv("winogender.csv")  # or your loaded DataFrame from JSON/TSV
                pairs = []

                # Group similar examples by prefix (before '.male.txt' etc.)
                df["group_id"] = df["sentid"].apply(lambda x: ".".join(x.split(".")[:-1]))

                for group_id, group in df.groupby("group_id"):
                    # Each group has male, female, and neutral versions
                    neutral = group[group["gender"] == "neutral"]
                    males = group[group["gender"] == "male"]
                    females = group[group["gender"] == "female"]

                    if len(neutral) == 0:
                        continue  # skip incomplete groups

                    neutral_row = neutral.iloc[0]
                    neutral_sentence = neutral_row["sentence"]
                    correct_target = neutral_row["target"]

                    # unbiased baseline answer
                    chosen = f"The pronoun '{neutral_row['pronoun']}' refers to the {correct_target}."
                    prompt = neutral_sentence

                    # reject gendered assumptions
                    for _, row in pd.concat([males, females]).iterrows():
                        rejected = f"The pronoun '{row['pronoun']}' refers to the {row['target']}."
                        pairs.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                            "participant": row["participant"],
                            "occupation": row["occupation"]
                        })

                pairs_df = pd.DataFrame(pairs)
                print(pairs_df.head())

                # Load tokenizer and base model
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(args.model_path)

                # Prepare data
                train_data = [{
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"]
                } for _, row in pairs_df.iterrows()]

                print("Starting DPO fine-tuning using TRLX...")

                from trl import DPOTrainer, DPOConfig
                from datasets import Dataset

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
                    report_to='none'
                )

                train_data = Dataset.from_list(train_data)

                def preprocess(example):
                    return {
                        "prompt_ids": tokenizer(
                            example["prompt"],
                            truncation=True,
                            padding="max_length",
                            max_length=256
                        )["input_ids"],
                        "chosen_ids": tokenizer(
                            example["chosen"],
                            truncation=True,
                            padding="max_length",
                            max_length=128
                        )["input_ids"],
                        "rejected_ids": tokenizer(
                            example["rejected"],
                            truncation=True,
                            padding="max_length",
                            max_length=128
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
    # quantize model
    model = quantize_model(model, args)


    # # save the quantized model
    # if args.output_path:
    #     model.save_pretrained(args.output_path, safe_serialization=False)
    #     enc.save_pretrained(args.output_path)


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
            output_path = "{}/{}/kv_{}_w_{}_a_{}.jsonl".format(task_name, args.model_path, args.kv_bit, args.w_bit, args.a_bit)
            print("* Output: ", output_path)
            if not os.path.exists("{}/{}".format(task_name, args.model_path)):
                os.makedirs("{}/{}".format(task_name, args.model_path))
            with open(output_path, 'w') as f:
                f.write(json.dumps(results['results'][task_name]) + "\n")



if __name__ == "__main__":
    main()
