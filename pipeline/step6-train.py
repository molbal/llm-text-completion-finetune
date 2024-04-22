import argparse

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="unsloth/llama-3-8b-bnb-4bit", required=False,
                        help="Base model to fine-tune. Default: unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to train on, e.g. 'molbal/horror-novel-chunks'")

    args = parser.parse_args()

    max_seq_length = 8192  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    dataset = load_dataset(args.dataset, split="train")
    EOS_TOKEN = tokenizer.eos_token
    print("EOS Token:")
    print(EOS_TOKEN)

    def formatting_func(example):
        return example["chunk"] + EOS_TOKEN

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="chunk",
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        packing=True,  # Packs short sequences together to save time!
        formatting_func=formatting_func,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            num_train_epochs=5,
            learning_rate=2e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.1,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/output/",
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained_gguf("/output/gguf-quant/", tokenizer, quantization_method="q4_k_m")
    print(f"✅ Done.")

if __name__ == "__main__":
    main()
