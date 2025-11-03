import os
import json
import argparse
import datasets

from qllm_eval.quantization.quant_wrapper import quantize_model
from qllm_eval.utils import build_model_and_enc
from qllm_eval.evaluation.q_harness.lm_eval_adaptor import LMEvalAdaptor
from lm_eval import evaluator

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from trl import RewardTrainer, PPOTrainer, PPOConfig

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
parser.add_argument("--dataset_name", type=str, default="nyu-mll/crows-pairs", help="Dataset for all RLHF stages")
parser.add_argument("--rlhf_chosen", type=str, required=True)
parser.add_argument("--rlhf_rejected", type=str, required=True)
parser.add_argument("--rlhf_prompt", type=str, required=True)
args = parser.parse_args()


def main():
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    print("* Quantization Format: kv_{}_w_{}_a_{}".format(args.kv_bit, args.w_bit, args.a_bit))
    if "falcon" in args.model_path.lower():
        args.kv_group_size = 64
        args.w_group_size = 64

    # Load same model and tokenizer for all stages
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # === Load dataset once ===
    raw_dataset = datasets.load_dataset(args.dataset_name)
    dataset = raw_dataset["test"]

    # === Stage 1: SFT ===
    print("* Stage 1: Supervised Fine-Tuning")
    def tokenize_fn(batch):
        return tokenizer(batch["sent_less"], truncation=True, padding="max_length", max_length=256)
    tokenized_sft = dataset.map(tokenize_fn, batched=True)

    sft_args = TrainingArguments(
        output_dir="./sft_model",
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_total_limit=1,
    )

    sft_trainer = Trainer(model=model, args=sft_args, train_dataset=tokenized_sft)
    sft_trainer.train()
    print("* SFT completed")

    # === Stage 2: Reward Model ===
    print("* Stage 2: Reward Model Training using same LM")
    pairs = [{"prompt": row[args.prompt], "chosen": row[args.chosen], "rejected": row[args.rejected]} for row in dataset]

    reward_trainer = RewardTrainer(
        model=model,  # same LM used as reward model
        tokenizer=tokenizer,
        train_dataset=pairs,
        args=None,
    )
    reward_trainer.train()
    print("* Reward model training completed")

    # === Stage 3: RLHF PPO ===
    print("* Stage 3: PPO alignment using same LM as reward")
    ppo_config = PPOConfig(batch_size=2, forward_batch_size=1)
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,        # same LM
        ref_model=None,
        tokenizer=tokenizer,
        reward_model=model, # same LM as reward
        dataset=pairs,
    )
    ppo_trainer.train()
    model.save_pretrained("./aligned_model")
    print("* RLHF alignment complete")

    # === Quantization ===
    model, enc = build_model_and_enc("./aligned_model", args.use_flash_attn, args.kv_bit, args.kv_group_size)
    model = quantize_model(model, args)

    # === Evaluation ===
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
        for task_name in task_names:
            output_path = f"{task_name}/{args.model_path}/kv_{args.kv_bit}_w_{args.w_bit}_a_{args.a_bit}.jsonl"
            print("* Output: ", output_path)
            os.makedirs(f"{task_name}/{args.model_path}", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json.dumps(results["results"][task_name]) + "\n")


if __name__ == "__main__":
    main()
