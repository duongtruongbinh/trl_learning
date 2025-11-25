"""Model helpers for Qwen-based DPO runs."""
from __future__ import annotations

from typing import Any, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional Unsloth acceleration
    from unsloth import FastLanguageModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FastLanguageModel = None  # type: ignore


def load_policy_and_tokenizer(
    model_name: str,
    use_unsloth: bool = False,
    **model_kwargs: Any,
) -> Tuple[Any, Any]:
    """Return a policy model and tokenizer/processor."""

    if use_unsloth and FastLanguageModel:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name, **model_kwargs)
        model = FastLanguageModel.get_peft_model(model)
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer
