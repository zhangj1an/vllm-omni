from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import AutoWeightsLoader

_SELF_ATTN_QKV = ".self_attn.to_qkv.weight"
_CROSS_ATTN_KV = ".cross_attn.to_kv.weight"
_MODEL_PRETRANSFORM_PREFIX_MAP = (
    ("_model.pretransform.model.", "_model.pretransform."),
    (
        "_model.conditioner.conditioners.audio_prompt.pretransform.model.",
        "_model.conditioner.conditioners.audio_prompt.pretransform.",
    ),
)


def remap_audiox_split_fused_attention_linears(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Replace fused ``to_qkv`` / ``to_kv`` tensors with split ``to_q``/``to_k``/``to_v`` shards."""
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
    """``in_proj`` is stacked [W_q; W_k; W_v] with shape (3*D, D). Same for bias (3*D,)."""
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
    """Return a new weight list suitable for :class:`~vllm_omni.diffusion.models.audiox.audiox_maf.MAF_Block`."""
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
    """True for official AudioX Oobleck+VAE+Snake pretransform JSON."""
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


def should_remap_audiox_vae_to_diffusers(model_config: dict[str, Any] | None) -> bool:
    if model_config is None:
        return False
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
    model_config: dict[str, Any] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Rewrite VAE tensor names for Diffusers Oobleck under ``pretransform.inner``."""
    d: dict[str, torch.Tensor] = dict(weights)
    if not should_remap_audiox_vae_to_diffusers(model_config):
        return list(d.items())

    m = model_config or {}
    inner_model = m.get("model") if isinstance(m, dict) else None
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
    model_config: dict | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Apply AudioX checkpoint remapping rules in one place."""
    remapped = remap_audiox_maf_weights(weights)
    remapped = remap_audiox_oobleck_weights_for_diffusers(remapped, model_config=model_config)
    remapped = remap_audiox_split_fused_attention_linears(remapped)
    return remapped


def filter_unused_keys(weights: Iterable[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    """Reserved hook for explicit ignored-key policy during load."""
    return list(weights)


def load_audiox_weights(
    pipeline: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    model_config: dict | None = None,
) -> None:
    loader = AutoWeightsLoader(pipeline)
    remapped = remap_audiox_state_dict(weights, model_config=model_config)
    remapped = filter_unused_keys(remapped)
    loader.load_weights(remapped)
