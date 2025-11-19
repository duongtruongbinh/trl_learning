import time
import torch
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig

# 1. Cấu hình Quantization (4-bit) để tiết kiệm VRAM tối đa cho việc Generation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# 2. Cấu hình LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Cấu hình GRPO
training_args = GRPOConfig(
    output_dir="./results_grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,      
    learning_rate=5e-6,               # GRPO thường cần LR nhỏ hơn SFT
    bf16=True,
    logging_steps=1,
    max_completion_length=512,        # Giới hạn độ dài câu trả lời sinh ra
    num_generations=4,                # Quan trọng: Chỉ sinh 4 mẫu mỗi prompt để tránh OOM (Mặc định thường là 8)
    report_to="none",
)

# Hàm reward đơn giản (như ví dụ của bạn)
def reward_function(completions, **kwargs):
    return [float(len(set(completion.lower()))) for completion in completions]

# Khởi tạo Trainer
# Lưu ý: Model path trỏ đến thư mục model 1.7B của bạn
trainer = GRPOTrainer(
    model="/mnt/dataset1/pretrained_fm/Qwen_Qwen3-1.7B",
    reward_funcs=reward_function,
    train_dataset=load_dataset("trl-lib/tldr", split="train"),
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