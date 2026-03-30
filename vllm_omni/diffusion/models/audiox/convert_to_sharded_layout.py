# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Convert one legacy AudioX bundle shape (``config.json`` + ``model.ckpt``) into vLLM-Omni sharded layout.

Produces the single Diffusers-style component layout expected by ``AudioXPipeline``
(per-subfolder safetensors + ``DiffusersPipelineLoader``).

Example::

    python -m vllm_omni.diffusion.models.audiox.convert_to_sharded_layout \\
        --input-dir /path/to/audiox_weights_base \\
        --output-dir /path/to/audiox_weights_sharded

The input directory must contain ``config.json`` and ``model.ckpt``.

Output layout::

    config.json  (copied)
    model_index.json
    transformer/diffusion_pytorch_model.safetensors
    conditioners/diffusion_pytorch_model.safetensors
    vae/diffusion_pytorch_model.safetensors   # only if checkpoint contains ``pretransform.*`` keys
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch
from collections.abc import Mapping

from vllm_omni.diffusion.models.audiox.sharded_weights import (
    AUDIOX_WEIGHT_LAYOUT_SHARDED,
    CONDITIONERS_SAFETENSORS,
    TRANSFORMER_SAFETENSORS,
    VAE_SAFETENSORS,
)

try:
    from safetensors.torch import save_file as safetensors_save_file
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `safetensors` package is required for this script. Install it (e.g. `pip install safetensors`)."
    ) from e


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
    """safetensors rejects non-contiguous tensors (common for views after torch.load)."""
    return {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}


def _resolve_ckpt_path(input_dir: str) -> str:
    ckpt = os.path.join(os.path.abspath(input_dir), "model.ckpt")
    if os.path.isfile(ckpt):
        return ckpt
    raise FileNotFoundError(f"No checkpoint found under {input_dir} (expected model.ckpt).")


def _load_legacy_checkpoint_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise RuntimeError(f"Expected dict checkpoint at {ckpt_path}, got {type(loaded)}")

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
    state = _load_legacy_checkpoint_state_dict(ckpt_path)
    transformer_sd, conditioners_sd, vae_sd = _partition_keys(state)

    t_dir = os.path.join(output_dir, "transformer")
    c_dir = os.path.join(output_dir, "conditioners")
    os.makedirs(t_dir, exist_ok=True)
    os.makedirs(c_dir, exist_ok=True)

    safetensors_save_file(_contiguous_state_dict(transformer_sd), os.path.join(output_dir, TRANSFORMER_SAFETENSORS))
    safetensors_save_file(_contiguous_state_dict(conditioners_sd), os.path.join(output_dir, CONDITIONERS_SAFETENSORS))
    if vae_sd:
        v_dir = os.path.join(output_dir, "vae")
        os.makedirs(v_dir, exist_ok=True)
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
