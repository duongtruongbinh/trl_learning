# GRPO Config README

This README explains how to configure **GRPO training** using the `GRPOConfig` class (which extends `transformers.TrainingArguments`).

---

## 1. Example config structure

A typical `config/grpo.yaml` might look like this:

    output_dir: "models/checkpoints"

    training:
      per_device_train_batch_size: 2
      num_generations: 8
      max_prompt_length: 128
      max_completion_length: 256
      learning_rate: 1e-6
      logging_steps: 10
      num_train_epochs: 1
      beta: 0.0
      loss_type: "dapo"
      scale_rewards: "group"
      bf16: true

    model:
      name: "Qwen/Qwen2-0.5B-Instruct"
      model_init_kwargs:
        torch_dtype: "bfloat16"

    dataset:
      train_path: "data/raw/train.jsonl"
      val_path: "data/raw/test.jsonl"

In Python you would typically load and use it like:

    cfg = load_config("config/grpo.yaml")

    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        **cfg["training"],           # all GRPOConfig/TrainingArguments fields
    )

    model_name = cfg["model"]["name"]
    model_init_kwargs = cfg["model"].get("model_init_kwargs", {})

    train_path = cfg["dataset"]["train_path"]
    val_path = cfg["dataset"]["val_path"]

---

## 2. Main groups of `GRPOConfig` parameters

All fields below live in the `training:` section of your YAML (unless stated otherwise) and are passed into `GRPOConfig`.

### 2.1. Parameters overriding `TrainingArguments`

These come from `transformers.TrainingArguments`, but `GRPOConfig` overrides the defaults:

- **`learning_rate: float = 1e-6`**  
  Initial LR for AdamW.  
  GRPO training usually prefers smaller LR, especially for larger models.

- **`logging_steps: int | float = 10`**  
  How often to log during training.  
  - Integer → number of steps.  
  - Float in `[0, 1)` → fraction of total training steps.

- **`gradient_checkpointing: bool = True`**  
  - `True` → save GPU memory with gradient checkpointing, slower backward pass.  
  - `False` → faster but uses more memory.

- **`bf16: bool | None`**  
  - If `None`, it is set to `not fp16`.  
  - On modern GPUs (Ampere+), `bf16=True` is often preferred over `fp16`.

- **`lr_scheduler_kwargs: dict | str | None`**  
  Extra arguments passed to the LR scheduler, for example:

      lr_scheduler_kwargs:
        num_cycles: 1

> All standard `TrainingArguments` fields still work (e.g. `per_device_train_batch_size`, `num_train_epochs`, `warmup_steps`, `weight_decay`, `optim`, `save_steps`, etc.). Only the defaults for a few were changed in `GRPOConfig`.

---

### 2.2. Model and reference model parameters

- **`model_init_kwargs: dict | str | None`**  
  Extra kwargs for `AutoModelForCausalLM.from_pretrained` when `model` is given as a string. Example:

      model_init_kwargs:
        torch_dtype: "bfloat16"
        trust_remote_code: true

- **`disable_dropout: bool = False`**  
  - `True` → disable dropout in the model.  
  - Useful when using a reference model to ensure stable log-probs.

- **`cast_lm_head_to_fp32: bool = False`**  
  - Casts the LM head of the policy + reference models to `float32`.  
  - Recommended (in ScaleRL) for models with **untied** word embeddings and LM head.

- **`beta: float = 0.0`**  
  KL coefficient. Controls use of a reference model:
  - `0.0` (default) → **no** reference model is loaded (faster, less memory).  
  - `> 0` → reference model + KL penalty are used.

- **`sync_ref_model: bool = False`**  
  - `True` → enable periodic synchronization of the reference model with the current policy (TR-DPO-style).

- **`ref_model_mixup_alpha: float = 0.6`**  
  Mix coefficient used when `sync_ref_model=True`:

  \[
  \pi_\text{ref} \leftarrow \alpha \pi_\theta + (1 - \alpha)\pi_{\text{ref, prev}}
  \]

- **`ref_model_sync_steps: int = 512`**  
  Number of steps between reference model syncs (τ in TR-DPO).

---

### 2.3. Data preprocessing

- **`remove_unused_columns: bool = False`**  
  GRPO typically relies on extra columns (e.g. `solution`, `meta`), so default is `False`.  
  - `True` → keep only `"prompt"`. Only use if your reward function does not need other fields.

- **`max_prompt_length: int | None = 512`**  
  Maximum prompt length after tokenization.  
  - If longer, the sequence is **left-truncated** (keep the end).

- **`num_generations: int | None = 8`**  
  Number of sampled completions **per prompt**.  
  - Must be **≥ 2** (required by GRPO).  
  - Impacts memory and generation time.

- **`max_completion_length: int | None = 256`**  
  Maximum length of generated completions.

- **`ds3_gather_for_generation: bool = True`**  
  DeepSpeed ZeRO-3 only:
  - `True` → gather model weights for generation, faster generation.  
  - `False` → can train models larger than a single GPU’s VRAM but generation is slower; not compatible with vLLM.

- **`shuffle_dataset: bool = True`**  
  Shuffle the training dataset.

---

### 2.4. Generation (sampling) parameters

These affect how completions are generated during GRPO.

#### Batch shape / frequency

- **`generation_batch_size: int | None`**  
- **`steps_per_generation: int | None`**

These two are **mutually exclusive**: they cannot be set at the same time.

The `__post_init__` logic:

1. If **both are `None`**:

       steps_per_generation = gradient_accumulation_steps
       generation_batch_size = (
           per_device_train_batch_size * world_size * steps_per_generation
       )

2. If `generation_batch_size` is set, `steps_per_generation` is `None`:
   - `generation_batch_size` **must be divisible** by  
     `per_device_train_batch_size * world_size`.
   - Then:

         steps_per_generation = generation_batch_size // (
             per_device_train_batch_size * world_size
         )

3. If `steps_per_generation` is set, `generation_batch_size` is `None`:

       generation_batch_size = (
           per_device_train_batch_size * world_size * steps_per_generation
       )

If both are set at once → error.

Also, `generation_batch_size` **must be divisible** by `num_generations`
(we need full prompt groups with no partial batches).

#### Sampling controls

- **`temperature: float = 1.0`**  
  Higher → more random sampling.

- **`top_p: float = 1.0`**  
  Nucleus sampling (0–1). `1.0` = no truncation.

- **`top_k: int | None = None`**  
  Keep only top-k tokens by probability. `None` = disabled.

- **`min_p: float | None = None`**  
  Minimum token probability (scaled by the most likely token prob). Typical range: `0.01–0.2`.

- **`generation_kwargs: dict | None`**  
  Extra arguments for `GenerationConfig` (transformers) or `SamplingParams` (vLLM).  
  - If you include keys like `top_p`, `temperature`, etc., these **override** the main fields.

- **`chat_template_kwargs: dict | None`**  
  Extra arguments for `apply_chat_template` when formatting prompts for chat models.

- **`repetition_penalty: float = 1.0`**  
  > 1 → discourages repetition, < 1 → encourages repetition.

- **`use_transformers_paged: bool = False`**  
  Use the paged implementation in `transformers` for generation (only when `use_vllm=False`).

- **`cache_implementation: str | None`**  
  Selects the cache implementation for faster generation.

---

### 2.5. vLLM integration

- **`use_vllm: bool = False`**  
  - `True` → use vLLM for generation instead of `model.generate()`.

- **`vllm_mode: str = "server"`**  
  - `"server"`: GRPO trainer sends requests to a separate vLLM server (`trl vllm-serve`).  
  - `"colocate"`: vLLM runs in the same process and shares GPUs with training.

- **`vllm_model_impl: str = "vllm"`**  
  - `"vllm"`: use vLLM’s own implementation.  
  - `"transformers"`: wrap a transformers model inside vLLM.

- **`vllm_enable_sleep_mode: bool = False`**  
  Enable a “sleep” mode where vLLM sleeps during optimizer steps, then wakes up for sync/generation.

- **`vllm_guided_decoding_regex: str | None`**  
  Regex pattern for guided decoding with vLLM. `None` = disabled.

#### vLLM server parameters (`vllm_mode="server"`)

- **`vllm_server_base_url: str | None`**  
  If set (e.g. `"http://localhost:8000"`), host/port are ignored.

- **`vllm_server_host: str = "0.0.0.0"`**  
- **`vllm_server_port: int = 8000`**  
- **`vllm_server_timeout: float = 240.0`**  
  How long to wait for the server to become ready before raising a `ConnectionError`.

#### vLLM colocated parameters (`vllm_mode="colocate"`)

- **`vllm_gpu_memory_utilization: float = 0.3`**  
  Fraction of GPU memory vLLM can use.

- **`vllm_tensor_parallel_size: int = 1`**  
  Tensor parallel size for vLLM.

#### Importance sampling correction

- **`vllm_importance_sampling_correction: bool = True`**  
  Apply Truncated Importance Sampling (TIS) to fix off-policy effects between vLLM and training backend.

- **`vllm_importance_sampling_cap: float = 2.0`**  
  Upper bound `C` on the importance sampling ratio for stability.

---

### 2.6. GRPO-specific training hyperparameters

These control how GRPO itself behaves.

- **`num_iterations: int = 1`**  
  Number of GRPO iterations per batch (μ in the GRPO algorithm).

- **`epsilon: float = 0.2`**  
  Clipping parameter ε for the GRPO/PPO-style loss.

- **`delta: float | None = None`**  
  - Enables **two-sided** GRPO loss when set (INTELLECT-2).  
  - Recommended `delta > 1 + epsilon`.  
  - `None` → standard GRPO clipping (one-sided).

- **`epsilon_high: float | None = None`**  
  Upper-bound ε for clipping.  
  - If `None`, defaults to the same as `epsilon`.  
  - DAPO recommends `0.28`.

- **`importance_sampling_level: str = "token"`**  
  Controls how importance ratios are computed:
  - `"token"`: per-token ratios (one weight per token).  
  - `"sequence"`: average over valid tokens → one ratio per sequence (GSPO suggests this can be more stable).

- **`reward_weights: list[float] | None`**  
  Per-reward weights. Length must match the number of reward functions.  
  - `None` → all rewards weight = `1.0`.

- **`scale_rewards: str = "group"`**  
  Strategy for reward scaling:
  - `"group"` / `True`: scale by std **within each prompt group** (default).  
  - `"batch"`: scale by std across the **whole batch** (PPO Lite).  
  - `"none"` / `False`: no scaling (Dr. GRPO recommends this to avoid difficulty bias).

  Internally, `True` → `"group"`, `False` → `"none"`.

- **`loss_type: str = "dapo"`**  
  Choice of GRPO loss formulation:

  - `"grpo"`  
    Normalize by sequence length → can cause **length bias** (prefers shorter positives, longer negatives).  
    Not recommended.

  - `"dr_grpo"`  
    Normalize by a global constant equal to `max_completion_length` (Dr. GRPO) → removes length bias.

  - `"dapo"` (default)  
    Normalize by the number of active tokens in the **global accumulated batch** (DAPO) → removes length bias.

  - `"bnpo"`  
    Normalize by the number of active tokens in the **local batch**.  
    - Slight dependence on local batch size.  
    - When `per_device_train_batch_size == 1`, this reduces to original GRPO.

- **`mask_truncated_completions: bool = False`**  
  If `True`, completions that hit `max_completion_length` and are truncated are excluded from the loss, which can stabilize training.

- **`top_entropy_quantile: float = 1.0`**  
  ρ parameter from “Beyond the 80/20 Rule”:
  - Only keep the top-ρ quantile of tokens by entropy in the policy loss.  
  - Range `[0.0, 1.0]`.  
    - `0.0` → keep only the highest-entropy token.  
    - `1.0` → keep all tokens.  
  - Recommended around `0.2`.

- **`use_liger_loss: bool | None = None`**  
  Deprecated alias:
  - If set, a warning is emitted and `use_liger_kernel` is set accordingly.
  - **Constraint**: when `delta` is not `None`, you **cannot** use Liger kernel (two-sided GRPO loss is not supported).

---

### 2.7. Logging

- **`log_completions: bool = False`**  
  If `True`, log a sample of `(prompt, completion)` pairs every `logging_steps` steps:
  - If `rich` is installed → pretty-print to console.
  - If `wandb` / `trackio` logging enabled → log there as well.

- **`num_completions_to_print: int | None`**  
  Number of completions to print/log when `log_completions=True`.  
  - `None` → log all completions in the batch.

- **`wandb_log_unique_prompts: bool = False`**  
  - `True` → only unique prompts are logged to wandb.  
  - `False` → log all prompts.

---

## 3. Important validation constraints (`__post_init__`)

When editing configs, keep in mind that `__post_init__` will enforce several constraints:

1. **Generation batch vs `num_generations`**

   - `generation_batch_size % num_generations == 0` is required.  
   - The **global eval batch size** must also be divisible by `num_generations`:

         global_eval_bs = per_device_eval_batch_size * world_size
         assert global_eval_bs % num_generations == 0

2. **Minimum `num_generations`**

   - `num_generations < 2` will raise an error.  
   - GRPO requires at least 2 generations per prompt.

3. **Mutual exclusivity**

   - You **cannot** configure both `generation_batch_size` and `steps_per_generation` at the same time.

4. **Liger + two-sided loss**

   - If `use_liger_kernel` (through `use_liger_loss`) and `delta` is not `None`, an error is raised because Liger does not yet support two-sided GRPO loss.

---

## 4. Recommended minimal config template

As a starting point, a minimal `training:` section might look like:

    training:
      per_device_train_batch_size: 2
      num_train_epochs: 1
      learning_rate: 1e-6
      logging_steps: 10

      max_prompt_length: 128
      max_completion_length: 64

      num_generations: 8
      beta: 0.0
      loss_type: "dapo"
      scale_rewards: "group"

      bf16: true
      gradient_checkpointing: true

You can then gradually add more advanced settings as needed:

- Enable vLLM (`use_vllm`, configure server or colocate mode).
- Enable reference model sync (`sync_ref_model`, `ref_model_sync_steps`, `ref_model_mixup_alpha`).
- Experiment with `top_entropy_quantile`, `importance_sampling_level`, `scale_rewards`, and different `loss_type` values.

---
