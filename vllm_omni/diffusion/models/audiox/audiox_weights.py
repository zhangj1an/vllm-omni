"""Load and remap AudioX sharded weights for vLLM-Omni.

Originally, examples and checkpoints were associated with the Hugging Face org
``HKUSTAudio/AudioX`` (and related ``HKUSTAudio/AudioX-*`` repos). To align with the
on-disk layout and remap path implemented here, documentation and download instructions
currently use ``zhangj1an/AudioX`` instead.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

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

_SELF_ATTN_QKV = ".self_attn.to_qkv.weight"
_CROSS_ATTN_KV = ".cross_attn.to_kv.weight"
_MODEL_PRETRANSFORM_PREFIX_MAP = (
    ("_model.pretransform.model.", "_model.pretransform."),
    (
        "_model.conditioner.conditioners.audio_prompt.pretransform.model.",
        "_model.conditioner.conditioners.audio_prompt.pretransform.",
    ),
)


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
    model_config: dict[str, Any],
) -> list[DiffusersPipelineLoader.ComponentSource]:
    root = os.path.abspath(model_root)
    rev = getattr(od_config, "revision", None)

    def _src(allow_rel: str) -> DiffusersPipelineLoader.ComponentSource:
        return DiffusersPipelineLoader.ComponentSource(
            model_or_path=root,
            subfolder=None,
            revision=rev,
            prefix="_model.",
            fall_back_to_pt=False,
            allow_patterns_overrides=[allow_rel],
        )

    sources: list[DiffusersPipelineLoader.ComponentSource] = [
        _src(TRANSFORMER_SAFETENSORS),
        _src(CONDITIONERS_SAFETENSORS),
    ]
    return sources


def remap_audiox_split_fused_attention_linears(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    out: dict[str, torch.Tensor] = {}
    for k, v in weights:
        if k.endswith(_SELF_ATTN_QKV):
            base = k[: -len(_SELF_ATTN_QKV)]
            wq, wk, wv = v.chunk(3, dim=0)
            out[base + ".self_attn.to_q.weight"] = wq.contiguous()
            out[base + ".self_attn.to_k.weight"] = wk.contiguous()
            out[base + ".self_attn.to_v.weight"] = wv.contiguous()
        elif k.endswith(_CROSS_ATTN_KV):
            base = k[: -len(_CROSS_ATTN_KV)]
            wk, wv = v.chunk(2, dim=0)
            out[base + ".cross_attn.to_k.weight"] = wk.contiguous()
            out[base + ".cross_attn.to_v.weight"] = wv.contiguous()
        else:
            out[k] = v
    return list(out.items())


def remap_audiox_merge_cross_attn_to_kv_for_upstream_dit(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    d = dict(weights)
    drop: set[str] = set()
    add: dict[str, torch.Tensor] = {}
    for k in list(d.keys()):
        if not k.endswith(".cross_attn.to_k.weight"):
            continue
        base = k[: -len("to_k.weight")]
        vk = base + "to_v.weight"
        if vk not in d:
            continue
        wk, wv = d[k], d[vk]
        add[base + "to_kv.weight"] = torch.cat([wk, wv], dim=0).contiguous()
        drop.add(k)
        drop.add(vk)
    if not add:
        return list(d.items())
    out: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if k in drop:
            continue
        out[k] = v
    out.update(add)
    return list(out.items())


def _split_cross_in_proj(
    w: torch.Tensor, b: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    wq, wk, wv = w.chunk(3, dim=0)
    to_kv_w = torch.cat([wk, wv], dim=0)
    if b is None:
        return wq, to_kv_w, None, None
    bq, bk, bv = b.chunk(3, dim=0)
    to_kv_b = torch.cat([bk, bv], dim=0)
    return wq, to_kv_w, bq, to_kv_b


def remap_audiox_maf_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    d = dict(weights)
    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    cross_w_key = cross_b_key = None
    for k in d:
        if k.endswith("maf_block.cross_attn.in_proj_weight"):
            cross_w_key = k
        elif k.endswith("maf_block.cross_attn.in_proj_bias"):
            cross_b_key = k

    if cross_w_key is not None:
        prefix = cross_w_key[: -len("cross_attn.in_proj_weight")]
        w = d[cross_w_key]
        b = d.get(cross_b_key) if cross_b_key is not None else None
        wq, wkv, bq, bkv = _split_cross_in_proj(w, b)
        out[prefix + "cross_block.to_q.weight"] = wq.contiguous()
        out[prefix + "cross_block.to_kv.weight"] = wkv.contiguous()
        if bq is not None:
            out[prefix + "cross_block.to_q.bias"] = bq.contiguous()
        if bkv is not None:
            out[prefix + "cross_block.to_kv.bias"] = bkv.contiguous()
        consumed.add(cross_w_key)
        if cross_b_key is not None:
            consumed.add(cross_b_key)

    for k in list(d.keys()):
        if k in consumed:
            continue
        if k.endswith("maf_block.cross_attn.out_proj.weight"):
            prefix = k[: -len("cross_attn.out_proj.weight")]
            out[prefix + "cross_block.to_out.weight"] = d[k].contiguous()
            consumed.add(k)
        elif k.endswith("maf_block.cross_attn.out_proj.bias"):
            prefix = k[: -len("cross_attn.out_proj.bias")]
            out[prefix + "cross_block.to_out.bias"] = d[k].contiguous()
            consumed.add(k)

    layer_re = re.compile(
        r"^(?P<prefix>.*maf_block\.)fusion_transformer\.layers\.(?P<idx>\d+)\.self_attn\.(?P<tail>.*)$"
    )

    for k, v in d.items():
        if k in consumed:
            continue
        m = layer_re.match(k)
        if m:
            prefix, idx, tail = m.group("prefix"), m.group("idx"), m.group("tail")
            if tail == "in_proj_weight":
                nk = f"{prefix}fusion_blocks.{idx}.to_qkv.weight"
            elif tail == "in_proj_bias":
                nk = f"{prefix}fusion_blocks.{idx}.to_qkv.bias"
            elif tail.startswith("out_proj."):
                nk = f"{prefix}fusion_blocks.{idx}.out_proj.{tail.split('.', 1)[1]}"
            else:
                nk = None
            if nk is not None:
                out[nk] = v.contiguous()
                consumed.add(k)
                continue

        if "maf_block.fusion_transformer.layers." in k:
            nk = k.replace("fusion_transformer.layers.", "fusion_blocks.", 1)
            nk = nk.replace(".linear1.", ".ff.0.")
            nk = nk.replace(".linear2.", ".ff.2.")
            out[nk] = v.contiguous()
            consumed.add(k)
            continue

        m_ff = re.match(
            r"^(?P<pre>.*maf_block\.fusion_blocks\.\d+)\.linear(?P<which>[12])\.(?P<tail>weight|bias)$",
            k,
        )
        if m_ff:
            ff_slot = "0" if m_ff.group("which") == "1" else "2"
            nk = f"{m_ff.group('pre')}.ff.{ff_slot}.{m_ff.group('tail')}"
            out[nk] = v.contiguous()
            continue

        out[k] = v

    return list(out.items())


def audiox_oobleck_ae_config_supported(cfg: dict[str, Any]) -> bool:
    if cfg.get("encoder", {}).get("type") != "oobleck" or cfg.get("decoder", {}).get("type") != "oobleck":
        return False
    if (cfg.get("bottleneck") or {}).get("type") != "vae":
        return False
    ec = cfg.get("encoder", {}).get("config") or {}
    dc = cfg.get("decoder", {}).get("config") or {}
    if not ec.get("use_snake", False):
        return False
    if dc.get("use_nearest_upsample", False):
        return False
    return True


def should_remap_audiox_vae_to_diffusers(model_config: dict[str, Any]) -> bool:
    m = model_config.get("model")
    if not isinstance(m, dict):
        raise ValueError("AudioX model config must contain a dict at key 'model'.")
    pt = m.get("pretransform")
    if not isinstance(pt, dict):
        raise ValueError("AudioX model config must contain a dict at model.pretransform.")
    ptc = pt.get("config")
    if not isinstance(ptc, dict):
        raise ValueError("AudioX model config must contain a dict at model.pretransform.config.")
    if not audiox_oobleck_ae_config_supported(ptc):
        raise ValueError(
            "AudioX inference-only weights remap requires Oobleck/VAE pretransform config "
            "(oobleck encoder+decoder, vae bottleneck, use_snake=true, use_nearest_upsample=false)."
        )
    return True


def _reshape_snake_param(tensor: torch.Tensor, *, target_ndim: int) -> torch.Tensor:
    if tensor.ndim == 1 and target_ndim == 3:
        return tensor.reshape(1, -1, 1)
    if tensor.ndim == 3 and target_ndim == 1:
        return tensor.reshape(-1)
    return tensor


def _map_encoder_decoder_suffix(
    suffix: str,
    *,
    num_blocks: int,
) -> tuple[str, bool] | None:
    m = re.fullmatch(r"encoder\.layers\.0\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"encoder.conv1.{m.group(1)}", False

    for blk in range(num_blocks):
        base = blk + 1
        bprefix = rf"encoder\.layers\.{base}\.layers"
        for ru in range(3):
            ru_name = f"res_unit{ru + 1}"
            inner_ru = ru
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.0\.(alpha|beta)", suffix)
            if m:
                return f"encoder.block.{blk}.{ru_name}.snake1.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.1\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"encoder.block.{blk}.{ru_name}.conv1.{m.group(1)}", False
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.2\.(alpha|beta)", suffix)
            if m:
                return f"encoder.block.{blk}.{ru_name}.snake2.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.3\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"encoder.block.{blk}.{ru_name}.conv2.{m.group(1)}", False
        m = re.fullmatch(rf"{bprefix}\.3\.(alpha|beta)", suffix)
        if m:
            return f"encoder.block.{blk}.snake1.{m.group(1)}", True
        m = re.fullmatch(rf"{bprefix}\.4\.(weight_g|weight_v|bias)", suffix)
        if m:
            return f"encoder.block.{blk}.conv1.{m.group(1)}", False

    tail_act = num_blocks + 1
    m = re.fullmatch(rf"encoder\.layers\.{tail_act}\.(alpha|beta)", suffix)
    if m:
        return f"encoder.snake1.{m.group(1)}", True
    m = re.fullmatch(rf"encoder\.layers\.{tail_act + 1}\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"encoder.conv2.{m.group(1)}", False

    m = re.fullmatch(r"decoder\.layers\.0\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"decoder.conv1.{m.group(1)}", False

    for blk in range(num_blocks):
        base = blk + 1
        bprefix = rf"decoder\.layers\.{base}\.layers"
        m = re.fullmatch(rf"{bprefix}\.0\.(alpha|beta)", suffix)
        if m:
            return f"decoder.block.{blk}.snake1.{m.group(1)}", True
        m = re.fullmatch(rf"{bprefix}\.1\.(weight_g|weight_v|bias)", suffix)
        if m:
            return f"decoder.block.{blk}.conv_t1.{m.group(1)}", False
        for ru in range(3):
            ru_name = f"res_unit{ru + 1}"
            inner_ru = 2 + ru
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.0\.(alpha|beta)", suffix)
            if m:
                return f"decoder.block.{blk}.{ru_name}.snake1.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.1\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"decoder.block.{blk}.{ru_name}.conv1.{m.group(1)}", False
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.2\.(alpha|beta)", suffix)
            if m:
                return f"decoder.block.{blk}.{ru_name}.snake2.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.3\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"decoder.block.{blk}.{ru_name}.conv2.{m.group(1)}", False

    tail0 = num_blocks + 1
    m = re.fullmatch(rf"decoder\.layers\.{tail0}\.(alpha|beta)", suffix)
    if m:
        return f"decoder.snake1.{m.group(1)}", True
    m = re.fullmatch(rf"decoder\.layers\.{tail0 + 1}\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"decoder.conv2.{m.group(1)}", False

    return None


def remap_audiox_oobleck_weights_for_diffusers(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    model_config: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    d: dict[str, torch.Tensor] = dict(weights)
    should_remap_audiox_vae_to_diffusers(model_config)

    inner_model = model_config["model"]
    pt = inner_model.get("pretransform") if isinstance(inner_model, dict) else None
    ptc = pt.get("config") if isinstance(pt, dict) else None
    strides = (ptc.get("encoder", {}).get("config") or {}).get("strides") if isinstance(ptc, dict) else None
    if not isinstance(strides, list) or not strides:
        raise ValueError("AudioX pretransform encoder strides must be a non-empty list for Oobleck remap.")
    num_blocks = len(strides)

    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for k, v in d.items():
        matched_prefix = None
        target_prefix = None
        for src_prefix, dst_prefix in _MODEL_PRETRANSFORM_PREFIX_MAP:
            if k.startswith(src_prefix):
                matched_prefix = src_prefix
                target_prefix = dst_prefix
                break
        if matched_prefix is None:
            out[k] = v
            continue
        suffix = k[len(matched_prefix) :]
        if suffix.startswith("encoder.") or suffix.startswith("decoder."):
            mapped = _map_encoder_decoder_suffix(suffix, num_blocks=num_blocks)
            if mapped is None:
                unmapped.append(k)
                continue
            new_suffix, snake = mapped
            tensor = _reshape_snake_param(v, target_ndim=3) if snake else v
            out[target_prefix + new_suffix] = tensor
        else:
            out[k] = v

    if unmapped:
        raise ValueError(
            "AudioX Oobleck -> Diffusers VAE remap: unmapped pretransform tensor(s): "
            + ", ".join(unmapped[:12])
            + (" ..." if len(unmapped) > 12 else "")
        )
    return list(out.items())


def remap_audiox_state_dict(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    model_config: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    normalized: list[tuple[str, torch.Tensor]] = []
    for k, v in weights:
        if k.startswith("_model."):
            normalized.append((k, v))
        elif k.startswith(("conditioner.", "maf_block.", "pretransform.")):
            normalized.append(("_model." + k, v))
        else:
            normalized.append((k, v))
    remapped = remap_audiox_maf_weights(normalized)
    remapped = remap_audiox_oobleck_weights_for_diffusers(remapped, model_config=model_config)
    remapped = remap_audiox_split_fused_attention_linears(remapped)
    remapped = remap_audiox_merge_cross_attn_to_kv_for_upstream_dit(remapped)
    return remapped


def filter_unused_keys(weights: Iterable[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    return list(weights)


def load_audiox_weights(
    pipeline: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    model_config: dict[str, Any],
) -> None:
    loader = AutoWeightsLoader(pipeline)
    remapped = remap_audiox_state_dict(weights, model_config=model_config)
    remapped = filter_unused_keys(remapped)
    loader.load_weights(remapped)
