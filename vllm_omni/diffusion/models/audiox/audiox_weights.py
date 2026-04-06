"""Load canonical AudioX sharded weights for vLLM-Omni.

Weights must match the module state dict (e.g. ``zhangj1an/AudioX`` on Hugging Face).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

AUDIOX_WEIGHT_LAYOUT_SHARDED = "vllm_omni_component_sharded"

TRANSFORMER_SAFETENSORS = "transformer/diffusion_pytorch_model.safetensors"
CONDITIONERS_SAFETENSORS = "conditioners/diffusion_pytorch_model.safetensors"

# --- JSON config keys (checkpoint / bundle schema) for inference-only AudioX ---

AUDIOX_PRETRANSFORM_CONFIG_KEYS_ALLOWED: frozenset[str] = frozenset({"type", "config", "scale"})

AUDIOX_VIDEO_PROMPT_CONFIG_KEYS_ALLOWED: frozenset[str] = frozenset(
    {"output_dim", "project_out", "clip_model_name"}
)
AUDIOX_TEXT_PROMPT_CONFIG_KEYS_ALLOWED: frozenset[str] = frozenset(
    {"output_dim", "t5_model_name", "max_length", "project_out"}
)
AUDIOX_AUDIO_PROMPT_CONFIG_KEYS_ALLOWED: frozenset[str] = frozenset(
    {
        "output_dim",
        "latent_seq_len",
        "pretransform_ckpt_path",
        "mask_ratio_start",
        "mask_ratio_end",
    }
)


def validate_audiox_pretransform_config_keys(pretransform_config: Mapping[str, Any]) -> None:
    extra = set(pretransform_config) - AUDIOX_PRETRANSFORM_CONFIG_KEYS_ALLOWED
    if extra:
        raise ValueError(
            f"Unsupported pretransform config keys for AudioX inference: {sorted(extra)}"
        )


def resolve_pretransform_scale(pretransform_config: Mapping[str, Any], icfg: Any) -> float:
    return float(pretransform_config.get("scale", getattr(icfg, "scaling_factor", 1.0)))


def resolve_vae_latent_channels(icfg: Any) -> int:
    return int(getattr(icfg, "latent_channels", getattr(icfg, "decoder_input_channels", 1)))


def resolve_vae_audio_channels(icfg: Any) -> int:
    return int(getattr(icfg, "audio_channels", 2))


def _validate_subconfig_keys(
    *,
    label: str,
    cfg: Mapping[str, Any],
    allowed: frozenset[str],
) -> None:
    extra = set(cfg) - allowed
    if extra:
        raise ValueError(f"Unsupported {label} config keys for AudioX inference: {sorted(extra)}")


def prepare_audiox_video_text_conditioner_configs(
    *,
    cond_dim: int,
    video_prompt: Mapping[str, Any],
    text_prompt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge ``cond_dim`` as ``output_dim`` and keep only keys supported for weight loading / inference."""
    video_cfg: dict[str, Any] = {"output_dim": cond_dim, **dict(video_prompt)}
    text_cfg: dict[str, Any] = {"output_dim": cond_dim, **dict(text_prompt)}

    _validate_subconfig_keys(
        label="video_prompt",
        cfg=video_cfg,
        allowed=AUDIOX_VIDEO_PROMPT_CONFIG_KEYS_ALLOWED,
    )
    _validate_subconfig_keys(
        label="text_prompt",
        cfg=text_cfg,
        allowed=AUDIOX_TEXT_PROMPT_CONFIG_KEYS_ALLOWED,
    )

    video_out = {
        k: video_cfg[k]
        for k in AUDIOX_VIDEO_PROMPT_CONFIG_KEYS_ALLOWED
        if k in video_cfg and k != "clip_model_name"
    }
    text_out = {k: text_cfg[k] for k in AUDIOX_TEXT_PROMPT_CONFIG_KEYS_ALLOWED if k in text_cfg}
    return video_out, text_out


def filter_audio_prompt_config_after_pretransform_build(audio_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Call after ``_build_pretransform`` pops ``sample_rate`` and ``pretransform_config`` from ``audio_cfg``."""
    _validate_subconfig_keys(
        label="audio_prompt",
        cfg=audio_cfg,
        allowed=AUDIOX_AUDIO_PROMPT_CONFIG_KEYS_ALLOWED,
    )
    return {
        k: audio_cfg[k]
        for k in AUDIOX_AUDIO_PROMPT_CONFIG_KEYS_ALLOWED
        if k in audio_cfg and k != "pretransform_ckpt_path"
    }


AUDIOX_DIFFUSION_MODEL_CONFIG_KEYS_NOT_FOR_DIT: frozenset[str] = frozenset({"video_fps"})


def strip_diffusion_model_config_for_audiox_dit(diffusion_model_config: dict[str, Any]) -> dict[str, Any]:
    """Remove keys present in JSON but not accepted by ``MMDiffusionTransformer`` constructors."""
    out = dict(diffusion_model_config)
    for k in AUDIOX_DIFFUSION_MODEL_CONFIG_KEYS_NOT_FOR_DIT:
        out.pop(k, None)
    return out


def audio_conditioning_input_samples_from_model_config(model_config: dict[str, Any]) -> int | None:
    """Samples length for audio conditioning from bundle ``config.json`` (``latent_seq_len`` × downsampling)."""
    try:
        m = model_config.get("model")
        if not isinstance(m, dict):
            return None
        cond = m.get("conditioning")
        if not isinstance(cond, dict):
            return None
        for item in cond.get("configs", []):
            if not isinstance(item, dict) or item.get("id") != "audio_prompt":
                continue
            c = item.get("config")
            if not isinstance(c, dict):
                continue
            ls = c.get("latent_seq_len")
            pt = c.get("pretransform_config")
            ds = None
            if isinstance(pt, dict):
                ptc = pt.get("config")
                if isinstance(ptc, dict):
                    ds = ptc.get("downsampling_ratio")
            if ls is not None and ds is not None:
                return int(ls) * int(ds)
    except (TypeError, ValueError):
        return None
    return None


def _model_root_has_file(model_root: str, rel: str) -> bool:
    return os.path.isfile(os.path.join(os.path.abspath(model_root), rel))


def _require_index_value(index: Mapping[str, Any], key: str, expected: Any) -> None:
    got = index.get(key)
    if got != expected:
        raise ValueError(f"AudioX model_index.json must set {key}={expected!r}; got {got!r}.")


def _require_required_files(root: str, rel_paths: Iterable[str]) -> None:
    missing = [rel for rel in rel_paths if not _model_root_has_file(root, rel)]
    if missing:
        raise FileNotFoundError(
            f"AudioX sharded layout missing required files under {root}: "
            + ", ".join(repr(rel) for rel in missing)
            + "."
        )


def load_audiox_model_index(model_root: str) -> dict[str, Any]:
    root = os.path.abspath(model_root)
    index_path = os.path.join(root, "model_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"AudioX sharded layout requires model_index.json under {root}")
    with open(index_path, encoding="utf-8") as f:
        index: dict[str, Any] = json.load(f)
    return index


def resolve_audiox_bundle_paths(model_root: str) -> tuple[str, dict[str, Any]]:
    root = os.path.abspath(model_root)
    idx = load_audiox_model_index(root)
    _require_index_value(idx, "weight_layout", AUDIOX_WEIGHT_LAYOUT_SHARDED)
    _require_index_value(idx, "config", "config.json")
    _require_index_value(idx, "transformer_weights", TRANSFORMER_SAFETENSORS)
    _require_index_value(idx, "conditioners_weights", CONDITIONERS_SAFETENSORS)
    _require_required_files(root, (TRANSFORMER_SAFETENSORS, CONDITIONERS_SAFETENSORS))
    return os.path.join(root, "config.json"), idx


def load_audiox_bundle_config(model_root: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    root = os.path.abspath(model_root)
    config_path, idx = resolve_audiox_bundle_paths(root)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"AudioX config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        model_config: dict[str, Any] = json.load(f)
    return config_path, model_config, idx


def build_sharded_component_sources(
    *,
    model_root: str,
    od_config: OmniDiffusionConfig,
) -> list[Any]:
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    root = os.path.abspath(model_root)
    rev = getattr(od_config, "revision", None)

    def _src(allow_rel: str) -> Any:
        return DiffusersPipelineLoader.ComponentSource(
            model_or_path=root,
            subfolder=None,
            revision=rev,
            prefix="_model.",
            fall_back_to_pt=False,
            allow_patterns_overrides=[allow_rel],
        )

    return [
        _src(TRANSFORMER_SAFETENSORS),
        _src(CONDITIONERS_SAFETENSORS),
    ]


def filter_unused_keys(weights: Iterable[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    return list(weights)


def load_audiox_weights(
    pipeline: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    from vllm.model_executor.models.utils import AutoWeightsLoader

    loader = AutoWeightsLoader(pipeline)
    loader.load_weights(filter_unused_keys(weights))
