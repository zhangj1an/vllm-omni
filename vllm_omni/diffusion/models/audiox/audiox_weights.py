from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from safetensors.torch import save_file as safetensors_save_file
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

AUDIOX_WEIGHT_LAYOUT_SHARDED = "vllm_omni_component_sharded"

TRANSFORMER_SAFETENSORS = "transformer/diffusion_pytorch_model.safetensors"
CONDITIONERS_SAFETENSORS = "conditioners/diffusion_pytorch_model.safetensors"
VAE_SAFETENSORS = "vae/diffusion_pytorch_model.safetensors"

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


def load_audiox_model_index(model_root: str) -> dict[str, Any]:
    root = os.path.abspath(model_root)
    index_path = os.path.join(root, "model_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"AudioX sharded layout requires model_index.json under {root}")
    with open(index_path, encoding="utf-8") as f:
        index: dict[str, Any] = json.load(f)
    return index


def model_config_has_pretransform(model_config: dict[str, Any]) -> bool:
    m = model_config.get("model")
    if not isinstance(m, dict):
        return False
    return m.get("pretransform") is not None


def resolve_audiox_bundle_paths(model_root: str) -> tuple[str, dict[str, Any]]:
    root = os.path.abspath(model_root)
    idx = load_audiox_model_index(root)
    if idx.get("weight_layout") != AUDIOX_WEIGHT_LAYOUT_SHARDED:
        raise ValueError(
            f"AudioX model_index.json must declare weight_layout={AUDIOX_WEIGHT_LAYOUT_SHARDED!r}; "
            f"got {idx.get('weight_layout')!r}."
        )
    if idx.get("config") != "config.json":
        raise ValueError("AudioX model_index.json must set 'config' to 'config.json'.")
    if idx.get("transformer_weights") != TRANSFORMER_SAFETENSORS:
        raise ValueError(f"AudioX model_index.json must set transformer_weights={TRANSFORMER_SAFETENSORS!r}.")
    if idx.get("conditioners_weights") != CONDITIONERS_SAFETENSORS:
        raise ValueError(f"AudioX model_index.json must set conditioners_weights={CONDITIONERS_SAFETENSORS!r}.")
    if not _model_root_has_file(root, TRANSFORMER_SAFETENSORS) or not _model_root_has_file(
        root, CONDITIONERS_SAFETENSORS
    ):
        raise FileNotFoundError(
            f"AudioX sharded layout missing required files under {root}: "
            f"{TRANSFORMER_SAFETENSORS!r}, {CONDITIONERS_SAFETENSORS!r}."
        )
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
    if model_config_has_pretransform(model_config):
        if not _model_root_has_file(root, VAE_SAFETENSORS):
            raise FileNotFoundError(
                f"AudioX config includes pretransform but {VAE_SAFETENSORS} was not found under {root}. "
                "Convert the bundle with vllm_omni.diffusion.models.audiox.audiox_weights "
                "or add the vae shard."
            )
        sources.append(_src(VAE_SAFETENSORS))
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
        return False
    pt = m.get("pretransform")
    if not isinstance(pt, dict):
        return False
    ptc = pt.get("config")
    if not isinstance(ptc, dict):
        return False
    return audiox_oobleck_ae_config_supported(ptc)


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
        return f"inner.encoder.conv1.{m.group(1)}", False

    for blk in range(num_blocks):
        base = blk + 1
        bprefix = rf"encoder\.layers\.{base}\.layers"
        for ru in range(3):
            ru_name = f"res_unit{ru + 1}"
            inner_ru = ru
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.0\.(alpha|beta)", suffix)
            if m:
                return f"inner.encoder.block.{blk}.{ru_name}.snake1.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.1\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"inner.encoder.block.{blk}.{ru_name}.conv1.{m.group(1)}", False
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.2\.(alpha|beta)", suffix)
            if m:
                return f"inner.encoder.block.{blk}.{ru_name}.snake2.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.3\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"inner.encoder.block.{blk}.{ru_name}.conv2.{m.group(1)}", False
        m = re.fullmatch(rf"{bprefix}\.3\.(alpha|beta)", suffix)
        if m:
            return f"inner.encoder.block.{blk}.snake1.{m.group(1)}", True
        m = re.fullmatch(rf"{bprefix}\.4\.(weight_g|weight_v|bias)", suffix)
        if m:
            return f"inner.encoder.block.{blk}.conv1.{m.group(1)}", False

    tail_act = num_blocks + 1
    m = re.fullmatch(rf"encoder\.layers\.{tail_act}\.(alpha|beta)", suffix)
    if m:
        return f"inner.encoder.snake1.{m.group(1)}", True
    m = re.fullmatch(rf"encoder\.layers\.{tail_act + 1}\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"inner.encoder.conv2.{m.group(1)}", False

    m = re.fullmatch(r"decoder\.layers\.0\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"inner.decoder.conv1.{m.group(1)}", False

    for blk in range(num_blocks):
        base = blk + 1
        bprefix = rf"decoder\.layers\.{base}\.layers"
        m = re.fullmatch(rf"{bprefix}\.0\.(alpha|beta)", suffix)
        if m:
            return f"inner.decoder.block.{blk}.snake1.{m.group(1)}", True
        m = re.fullmatch(rf"{bprefix}\.1\.(weight_g|weight_v|bias)", suffix)
        if m:
            return f"inner.decoder.block.{blk}.conv_t1.{m.group(1)}", False
        for ru in range(3):
            ru_name = f"res_unit{ru + 1}"
            inner_ru = 2 + ru
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.0\.(alpha|beta)", suffix)
            if m:
                return f"inner.decoder.block.{blk}.{ru_name}.snake1.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.1\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"inner.decoder.block.{blk}.{ru_name}.conv1.{m.group(1)}", False
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.2\.(alpha|beta)", suffix)
            if m:
                return f"inner.decoder.block.{blk}.{ru_name}.snake2.{m.group(1)}", True
            m = re.fullmatch(rf"{bprefix}\.{inner_ru}\.layers\.3\.(weight_g|weight_v|bias)", suffix)
            if m:
                return f"inner.decoder.block.{blk}.{ru_name}.conv2.{m.group(1)}", False

    tail0 = num_blocks + 1
    m = re.fullmatch(rf"decoder\.layers\.{tail0}\.(alpha|beta)", suffix)
    if m:
        return f"inner.decoder.snake1.{m.group(1)}", True
    m = re.fullmatch(rf"decoder\.layers\.{tail0 + 1}\.(weight_g|weight_v|bias)", suffix)
    if m:
        return f"inner.decoder.conv2.{m.group(1)}", False

    return None


def remap_audiox_oobleck_weights_for_diffusers(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    model_config: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    d: dict[str, torch.Tensor] = dict(weights)
    if not should_remap_audiox_vae_to_diffusers(model_config):
        return list(d.items())

    inner_model = model_config["model"]
    pt = inner_model.get("pretransform") if isinstance(inner_model, dict) else None
    ptc = pt.get("config") if isinstance(pt, dict) else None
    strides = (ptc.get("encoder", {}).get("config") or {}).get("strides") if isinstance(ptc, dict) else None
    if not isinstance(strides, list) or not strides:
        return list(d.items())
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
    remapped = remap_audiox_maf_weights(weights)
    normalized: list[tuple[str, torch.Tensor]] = []
    for k, v in remapped:
        if k.startswith("_model.model.model."):
            k = "_model.model." + k[len("_model.model.model.") :]
        elif k.startswith("model.model."):
            k = "model." + k[len("model.model.") :]
        normalized.append((k, v))
    remapped = normalized
    remapped = remap_audiox_oobleck_weights_for_diffusers(remapped, model_config=model_config)
    remapped = remap_audiox_split_fused_attention_linears(remapped)
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


def _partition_keys(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], ...]:
    transformer: dict[str, torch.Tensor] = {}
    conditioners: dict[str, torch.Tensor] = {}
    vae: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    for k, v in state.items():
        if k.startswith("model."):
            transformer[k] = v
        elif k.startswith("pretransform."):
            vae[k] = v
        elif k.startswith("conditioner.") or k.startswith("maf_block."):
            conditioners[k] = v
        else:
            unknown.append(k)
    if unknown:
        raise ValueError(
            "Checkpoint contains keys that are not mapped to transformer / vae / conditioners "
            f"shards (expected prefixes model., pretransform., conditioner., maf_block.): {unknown[:20]}"
            + (" ..." if len(unknown) > 20 else "")
        )
    return transformer, conditioners, vae


def _contiguous_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}


def _resolve_ckpt_path(input_dir: str) -> str:
    ckpt = os.path.join(os.path.abspath(input_dir), "model.ckpt")
    if os.path.isfile(ckpt):
        return ckpt
    raise FileNotFoundError(f"No checkpoint found under {input_dir} (expected model.ckpt).")


def _load_checkpoint_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping) or "state_dict" not in loaded or not isinstance(loaded["state_dict"], Mapping):
        raise RuntimeError(
            "AudioX inference conversion expects model.ckpt to contain a top-level 'state_dict' mapping."
        )
    loaded = loaded["state_dict"]

    out: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if isinstance(tensor, torch.Tensor):
            out[str(name)] = tensor
    return out


def convert_audiox_bundle(input_dir: str, output_dir: str, *, copy_config: bool = True) -> None:
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cfg_src = os.path.join(input_dir, "config.json")
    if not os.path.isfile(cfg_src):
        raise FileNotFoundError(f"Missing config.json in {input_dir}")
    if copy_config:
        cfg_dst = os.path.join(output_dir, "config.json")
        if os.path.normpath(cfg_src) != os.path.normpath(cfg_dst):
            shutil.copy2(cfg_src, cfg_dst)

    ckpt_path = _resolve_ckpt_path(input_dir)
    state = _load_checkpoint_state_dict(ckpt_path)
    transformer_sd, conditioners_sd, vae_sd = _partition_keys(state)

    os.makedirs(os.path.join(output_dir, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "conditioners"), exist_ok=True)

    safetensors_save_file(_contiguous_state_dict(transformer_sd), os.path.join(output_dir, TRANSFORMER_SAFETENSORS))
    safetensors_save_file(_contiguous_state_dict(conditioners_sd), os.path.join(output_dir, CONDITIONERS_SAFETENSORS))
    if vae_sd:
        os.makedirs(os.path.join(output_dir, "vae"), exist_ok=True)
        safetensors_save_file(_contiguous_state_dict(vae_sd), os.path.join(output_dir, VAE_SAFETENSORS))

    index: dict[str, Any] = {
        "_class_name": "AudioXPipeline",
        "config": "config.json",
        "weight_layout": AUDIOX_WEIGHT_LAYOUT_SHARDED,
        "source_checkpoint": os.path.basename(ckpt_path),
        "transformer_weights": TRANSFORMER_SAFETENSORS,
        "conditioners_weights": CONDITIONERS_SAFETENSORS,
    }
    if vae_sd:
        index["vae_weights"] = VAE_SAFETENSORS
    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Legacy bundle with config.json + model.ckpt")
    p.add_argument("--output-dir", required=True, help="Directory to write sharded safetensors + index")
    p.add_argument(
        "--no-copy-config",
        action="store_true",
        help="Do not copy config.json (output dir must already contain a compatible config.json)",
    )
    args = p.parse_args()
    convert_audiox_bundle(args.input_dir, args.output_dir, copy_config=not args.no_copy_config)
    print(f"Wrote sharded AudioX layout to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
