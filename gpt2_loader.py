"""Utilities for loading GPT-2 checkpoints and extracting interpretable signals."""
from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
# Backward compatible alias used by app.py sidebar
CACHE_DIR = HF_CACHE_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "small": {
        "id": "gpt2",
        "display": "GPT-2 Small",
        "layers": 12,
        "heads": 12,
        "context": 1024,
        "params": "124M",
    },
    "medium": {
        "id": "gpt2-medium",
        "display": "GPT-2 Medium",
        "layers": 24,
        "heads": 16,
        "context": 1024,
        "params": "355M",
    },
    "large": {
        "id": "gpt2-large",
        "display": "GPT-2 Large",
        "layers": 36,
        "heads": 20,
        "context": 1024,
        "params": "774M",
    },
    "xl": {
        "id": "gpt2-xl",
        "display": "GPT-2 XL",
        "layers": 48,
        "heads": 25,
        "context": 1024,
        "params": "1.5B",
    },
}
HF_MIRROR_DEFAULT = "https://hf-mirror.com"

ProgressCallback = Optional[Callable[[str, int], None]]
_MODEL_CACHE: Dict[Tuple[str, Optional[str]], Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


@dataclass
class GenerationSettings:
    """User-driven parameters that control GPT-2 inference."""

    model_size: str = "small"
    max_new_tokens: int = 120
    temperature: float = 0.7
    top_k: int = 5
    attention_layers: List[int] = field(default_factory=lambda: [1, 6, 12])
    attention_heads: List[int] = field(default_factory=lambda: list(range(1, 13)))
    max_context_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        self.model_size = self.model_size.lower()
        if self.model_size not in MODEL_SPECS:
            raise ValueError(f"Unsupported model size: {self.model_size}")
        self.max_new_tokens = int(min(max(self.max_new_tokens, 0), 300))
        self.temperature = float(min(max(self.temperature, 0.1), 1.5))
        self.top_k = int(min(max(self.top_k, 1), 100))
        spec = MODEL_SPECS[self.model_size]
        max_heads = spec["heads"]
        self.attention_layers = sorted({layer for layer in self.attention_layers if layer > 0})
        self.attention_heads = sorted({head for head in self.attention_heads if 0 < head <= max_heads})
        if not self.attention_layers:
            self.attention_layers = [1]
        if not self.attention_heads:
            self.attention_heads = list(range(1, min(12, max_heads) + 1))
        if self.max_context_tokens is None:
            self.max_context_tokens = int(spec.get("context", 1024))


@dataclass
class TokenStepMetrics:
    """Stores statistics for each generated token."""

    position: int
    token: str
    token_id: int
    probability: float
    rank: int
    perplexity: float


@dataclass
class GenerationArtifacts:
    """Bundle of tensors and friendly metadata derived from a GPT-2 run."""

    prompt: str
    generated_text: str
    combined_text: str
    tokens: List[str]
    token_ids: List[int]
    attentions: Dict[int, np.ndarray]
    hidden_states: Dict[int, np.ndarray]
    token_metrics: List[TokenStepMetrics]
    model_size: str


def resolve_device() -> torch.device:
    """Pick the best available computation device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _notify(callback: ProgressCallback, stage: str, percent: int) -> None:
    if not callback:
        return
    try:
        callback(stage, max(0, min(100, int(percent))))
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Progress callback error: %s", exc)


def _normalize_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    trimmed = endpoint.strip()
    if not trimmed:
        return None
    return trimmed.rstrip("/")


def _clear_other_models(active_key: Tuple[str, Optional[str]]) -> None:
    """Free cached models other than the active key to save memory."""

    to_delete = [key for key in _MODEL_CACHE if key != active_key]
    for key in to_delete:
        model, _ = _MODEL_CACHE.pop(key)
        try:
            del model
        except Exception:  # pragma: no cover
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_cached_models() -> None:
    """Completely clear all cached models to release memory."""

    keys = list(_MODEL_CACHE.keys())
    for key in keys:
        model, _ = _MODEL_CACHE.pop(key)
        try:
            del model
        except Exception:  # pragma: no cover
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_components(
    model_size: str,
    hf_endpoint: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download (if needed) and initialize a GPT-2 checkpoint and tokenizer."""

    normalized_endpoint = _normalize_endpoint(hf_endpoint)
    cache_key = (model_size, normalized_endpoint)
    if cache_key in _MODEL_CACHE:
        _notify(progress_callback, "download", 100)
        return _MODEL_CACHE[cache_key]

    _notify(progress_callback, "download", 5)
    components = _load_model_components(
        model_size,
        normalized_endpoint,
        tried_mirror=False,
        progress_callback=progress_callback,
    )
    _MODEL_CACHE[cache_key] = components
    _clear_other_models(cache_key)
    _notify(progress_callback, "download", 90)
    _notify(progress_callback, "download", 100)
    return components


def _load_model_components(
    model_size: str,
    hf_endpoint: Optional[str],
    tried_mirror: bool,
    progress_callback: ProgressCallback = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    spec = MODEL_SPECS[model_size]
    LOGGER.info("Loading %s", spec["display"])
    previous_endpoint = os.environ.get("HF_ENDPOINT")
    if hf_endpoint is not None:
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
    try:
        _notify(progress_callback, "download", 20)
        tokenizer = AutoTokenizer.from_pretrained(spec["id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _notify(progress_callback, "download", 40)
        model_kwargs = dict(
            low_cpu_mem_usage=True,
            torch_dtype=_resolve_dtype(resolve_device()),
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec["id"],
                attn_implementation="eager",
                **model_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                spec["id"],
                **model_kwargs,
            )
        _notify(progress_callback, "download", 70)
    except OSError as exc:
        if tried_mirror:
            raise
        if hf_endpoint is not None:
            raise
        mirror = _normalize_endpoint(os.environ.get("HF_MIRROR") or HF_MIRROR_DEFAULT)
        LOGGER.warning(
            "Primary download failed (%s). Switching to Hugging Face mirror %s and retrying...",
            exc,
            mirror,
        )
        return _load_model_components(
            model_size,
            mirror,
            tried_mirror=True,
            progress_callback=progress_callback,
        )
    finally:
        if hf_endpoint is not None:
            if previous_endpoint is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = previous_endpoint

    model.eval()
    return model, tokenizer


def _format_tokens(tokenizer: AutoTokenizer, ids: torch.Tensor) -> List[str]:
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
    return [token.replace("Ġ", "▁") for token in tokens]


def _compute_token_metrics(
    tokenizer: AutoTokenizer,
    scores: List[torch.Tensor],
    generated_ids: torch.Tensor,
    start_position: int,
) -> List[TokenStepMetrics]:
    metrics: List[TokenStepMetrics] = []
    for idx, (logits, token_id_tensor) in enumerate(zip(scores, generated_ids)):
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
        token_id = token_id_tensor.item()
        prob = float(probs[token_id].item())
        rank = int((torch.sum(probs > probs[token_id]) + 1).item())
        perplexity = float(1.0 / max(prob, 1e-6))
        token = tokenizer.convert_ids_to_tokens(token_id)
        metrics.append(
            TokenStepMetrics(
                position=start_position + idx,
                token=token.replace("Ġ", "▁"),
                token_id=token_id,
                probability=prob,
                rank=rank,
                perplexity=perplexity,
            )
        )
    return metrics


def _to_numpy_layers(layer_tensors: Tuple[torch.Tensor, ...]) -> Dict[int, np.ndarray]:
    layer_map: Dict[int, np.ndarray] = {}
    for idx, tensor in enumerate(layer_tensors, start=1):
        squeezed = tensor.detach().cpu().squeeze(0)
        layer_map[idx] = squeezed.numpy()
    return layer_map


def run_generation(
    prompt: str,
    settings: GenerationSettings,
    hf_endpoint: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> GenerationArtifacts:
    """Generate text and collect interpretable tensors for visualization."""

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("输入不能为空。请输入需要分析的文本。")

    _notify(progress_callback, "download", 5)
    model, tokenizer = load_model_components(
        settings.model_size,
        hf_endpoint=hf_endpoint,
        progress_callback=progress_callback,
    )
    device = resolve_device()
    model.to(device)

    _notify(progress_callback, "inference", 5)
    encoded = tokenizer(
        normalized_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_context_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    _notify(progress_callback, "inference", 15)
    _notify(progress_callback, "inference", 30)
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    _notify(progress_callback, "inference", 55)
    sequences = generation.sequences[0]
    full_token_ids = sequences.detach().cpu()
    generated_ids = sequences[input_ids.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    combined_text = tokenizer.decode(full_token_ids, skip_special_tokens=True)

    token_metrics = _compute_token_metrics(
        tokenizer=tokenizer,
        scores=generation.scores,
        generated_ids=generated_ids,
        start_position=input_ids.shape[-1],
    )

    _notify(progress_callback, "inference", 70)
    with torch.no_grad():
        forward_outputs = model(
            sequences.unsqueeze(0),
            output_attentions=True,
            output_hidden_states=True,
        )

    _notify(progress_callback, "inference", 85)
    attentions = _to_numpy_layers(forward_outputs.attentions)
    hidden_states = _to_numpy_layers(forward_outputs.hidden_states[1:])

    tokens = _format_tokens(tokenizer, full_token_ids)

    _notify(progress_callback, "inference", 95)
    _notify(progress_callback, "inference", 100)
    return GenerationArtifacts(
        prompt=normalized_prompt,
        generated_text=generated_text.strip(),
        combined_text=combined_text.strip(),
        tokens=tokens,
        token_ids=full_token_ids.tolist(),
        attentions=attentions,
        hidden_states=hidden_states,
        token_metrics=token_metrics,
        model_size=settings.model_size,
    )


def describe_model(model_size: str) -> str:
    """Return a friendly description for showing in the UI."""

    spec = MODEL_SPECS.get(model_size.lower())
    if not spec:
        return "Unknown"
    return f"{spec['display']} · 层数 {spec['layers']} · 头数 {spec['heads']}"
