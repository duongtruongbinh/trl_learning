import time
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,      
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    learning_rate=2e-4,                 
    bf16=True,
    logging_steps=10,
    report_to="none",
    dataloader_num_workers=4,
    max_steps=100                       
)

trainer = SFTTrainer(
    model="/mnt/dataset1/pretrained_fm/Qwen_Qwen3-1.7B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
    args=training_args,
    peft_config=peft_config,
)

torch.cuda.empty_cache()
start_time = time.perf_counter()
train_result = trainer.train()
end_time = time.perf_counter()

total_time = end_time - start_time
total_flops = train_result.metrics.get("total_flos", 0.0)
tflops_per_sec = (total_flops / total_time) / 1e12 if total_time > 0 else 0

print(f"\n{'='*30}")
print(f"Training Time: {total_time:.4f} s")
print(f"Total FLOPs:   {total_flops:.4e}")
print(f"Performance:   {tflops_per_sec:.4f} TFLOPs/s")
print(f"{'='*30}\n")