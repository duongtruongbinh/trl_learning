# 📘 SFTTrainer Quick Reference

**SFTTrainer** (Supervised Fine-Tuning Trainer) là lớp chính trong thư viện `trl` dùng để tinh chỉnh LLM trên tập dữ liệu có hướng dẫn (instruction datasets).

## 🛠️ Cấu hình quan trọng (`SFTConfig`)

Sử dụng `SFTConfig` (kế thừa `TrainingArguments`) để kiểm soát quá trình huấn luyện.

### 1\. Batch Size & Gradient


| Tham số | Ý nghĩa & Định nghĩa |
| :--- | :--- |
| `per_device_train_batch_size` | **Micro-Batch Size:** Số lượng mẫu dữ liệu thực tế được nạp vào VRAM của **1 GPU** tại một thời điểm. <br> *Quyết định việc có bị OOM (tràn bộ nhớ) hay không.* |
| `gradient_accumulation_steps` | **Accumulation:** Số bước "chờ" để tích lũy gradient trước khi thực sự cập nhật trọng số (weight update). Giúp mô phỏng batch lớn trên GPU yếu. |
| **Effective / Total Batch Size** | **Kích thước Batch thực tế:** Số lượng mẫu dữ liệu mô hình "nhìn thấy" trước khi thực hiện **1 bước cập nhật trọng số (1 step)**. <br> **Công thức:** `Per_Device * Accumulation * Số lượng GPUs`. |

### 2\. 🧮 Cách tính toán Steps & Epochs (Training Dynamics)

Hiểu cách tính này giúp bạn ước lượng thời gian train và chọn `max_steps` phù hợp.

**Công thức cốt lõi:**
$$Steps\_Per\_Epoch = \frac{Total\_Dataset\_Size}{Effective\_Batch\_Size}$$

**Ví dụ minh họa:**
Giả sử bạn có cấu hình sau:

  * **Dataset:** 10,000 mẫu (samples).
  * 2 GPUs.
  * **Config:** `per_device_train_batch_size=4`, `gradient_accumulation_steps=8`.
  * **Mục tiêu:** Train trong 3 Epochs.

**Tính toán:**

1.  **Tính Effective Batch Size:**
    $$4 \text{ (mẫu/GPU)} \times 8 \text{ (accumulation)} \times 2 \text{ (GPUs)} = \textbf{64 mẫu/step}$$
    *(Nghĩa là mỗi lần model cập nhật trọng số, nó đã học từ 64 mẫu dữ liệu).*

2.  **Tính số bước trong 1 Epoch:**
    $$10,000 / 64 = 156.25 \rightarrow \textbf{157 steps} \text{ (làm tròn lên)}$$

3.  **Tổng số bước training (Total Steps):**
    $$157 \text{ steps} \times 3 \text{ epochs} = \textbf{471 steps}$$

> [\!TIP]
> **Nghĩa là**
> Nếu set `max_steps=1000` cho dataset trên, mô hình sẽ train khoảng **6.4 Epochs** ($1000 / 157$). 

### 3\. Tài nguyên & Tốc độ

| Tham số | Ý nghĩa & Khuyên dùng |
| :--- | :--- |
| `gradient_checkpointing` | `True`. Hy sinh tốc độ tính toán (chậm hơn \~20%) để giảm mạnh VRAM (lưu ít activation hơn). Bắt buộc với model lớn. |
| `bf16` | `True` (nếu GPU hỗ trợ Ampere trở lên). Tăng tốc và giảm bộ nhớ so với FP32, ổn định hơn FP16. |

### 4\. Chiến lược Train (Steps vs Epochs)

| Tham số | Ý nghĩa |
| :--- | :--- |
| `num_train_epochs` | Số lần model duyệt qua toàn bộ dataset. |
| `max_steps` | Số bước update weights tuyệt đối (sẽ ghi đè `num_train_epochs`). |

> [\!NOTE]
>
>   * **Dataset nhỏ (\< 10k):** Set `num_train_epochs = 3`.
>   * **Dataset lớn (\> 50k):** Set `max_steps`. Thường chỉ cần **1000 - 2000 steps** (bất kể bao nhiêu epoch) là model đã học tốt hướng dẫn (alignment). Train full epoch với dataset khổng lồ thường lãng phí và gây quên kiến thức (catastrophic forgetting).

### 5\. Learning Rate

| Tham số | Ý nghĩa & Khuyên dùng |
| :--- | :--- |
| `learning_rate` | Tốc độ học. <br> **QLoRA/LoRA:** `2e-4`. <br> **Full Fine-tune:** `1e-5` đến `2e-5`. |
| `lr_scheduler_type` | `"cosine"` (mượt mà giảm dần) hoặc `"constant_with_warmup"`. |
| `warmup_ratio` | `0.03` (3% tổng steps). Giúp model làm quen dữ liệu từ từ, tránh shock gradient đầu chu kỳ. |



## 📝 Định dạng dữ liệu (Dataset Format)

**1. Conversational (Chuẩn Chat - Khuyên dùng):**
Tự động áp dụng chat template.

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

**2. Instruction/Response (Cổ điển):**

```json
{"prompt": "User: Hello\nAssistant:", "completion": " Hi there!"}
```



## 🚀 Code mẫu (Minimal Snippet)

Đoạn code dưới đây setup để đạt **Effective Batch Size = 16** trên 1 GPU.

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Giả sử dataset có 1000 mẫu
dataset = load_dataset("trl-lib/Capybara", split="train").select(range(1000))

args = SFTConfig(
    output_dir="./qwen_finetuned",
    max_seq_length=2048,
    packing=True,                   # Gom data để train nhanh hơn
    # --- Cấu hình Batch Size ---
    per_device_train_batch_size=2,  # Mỗi lần GPU load 2 mẫu (để không OOM)
    gradient_accumulation_steps=8,  # Tích lũy 8 lần mới update
    # => Effective Batch Size = 2 * 8 * 1(GPU) = 16
    # --- Cấu hình Steps ---
    num_train_epochs=3,             # Tổng steps sẽ là: (1000 / 16) * 3 ~= 189 steps
    
    learning_rate=2e-4,             # LR cho LoRA
    logging_steps=10,
    bf16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=dataset,
    args=args,
)

trainer.train()
```