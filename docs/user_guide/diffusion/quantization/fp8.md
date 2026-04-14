# FP8 Quantization

## Overview

FP8 quantization converts BF16/FP16 weights to FP8 at model load time. No calibration or pre-quantized checkpoint needed.

Depending on the model, either most linear layers are quantized with a few **built-in** BF16 exceptions in code (`quant_config=None`), or you must add extra skips via `ignored_layers`. See the [per-model table](#supported-models).

Built-in BF16 is used where small linear layers drive **timestep conditioning**, **per-block modulation** (scale/shift/gate), **input embedders**, or the **final latent projection**. Quantizing those paths caused visible noise or color drift on Z-Image, Qwen-Image, and FLUX.1-dev ([PR #2728](https://github.com/vllm-project/vllm-omni/pull/2728)); they stay in full precision automatically—you do not name them in `ignored_layers`.

Beyond that, common user-controlled skips include **image-stream MLPs** (`img_mlp`) on Qwen-Image: they see shifting latent statistics and benefit from `ignored_layers` for best quality. **Attention projections** (`to_qkv`, `to_out`) and **text-stream MLPs** (`txt_mlp`) are usually fine in FP8 when modulation and embedders stay BF16.

## Configuration

1. **Python API**: set `quantization="fp8"`. To skip sensitive layers, use `quantization_config` with `ignored_layers`.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# All layers quantized
omni = Omni(model="<your-model>", quantization="fp8")

# Skip sensitive layers
omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

2. **CLI**: pass `--quantization fp8` and optionally `--ignored-layers`.

```bash
# All layers
python text_to_image.py --model <your-model> --quantization fp8

# Skip sensitive layers
python text_to_image.py --model <your-model> --quantization fp8 --ignored-layers "img_mlp"

# Online serving
vllm serve <your-model> --omni --quantization fp8
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` (no calibration) or `"static"` |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

The available `ignored_layers` names depend on the model architecture (e.g., `to_qkv`, `to_out`, `img_mlp`, `txt_mlp`). Consult the transformer source for your target model.

## Supported Models

| Model | HF Models | FP8 scope | `ignored_layers` (optional) |
|-------|-----------|-----------|------------------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Main blocks (attention + FFN) use FP8. **Always BF16 in code:** timestep MLP, per-block adaLN modulation linear, patch and caption embedders, final layer (modulation + `proj_out`). | None required for those paths |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Joint attention and MLPs can use FP8. **Always BF16 in code:** timestep MLP, per-block `img_mod` / `txt_mod` linears, `img_in` / `txt_in`, `norm_out.linear`, `proj_out`. | Still recommend `img_mlp` for quality |
| Flux | `black-forest-labs/FLUX.1-dev` | **Single-stream** blocks (`single_transformer_blocks`) use FP8. **Always BF16 in code:** all **dual-stream** blocks (`transformer_blocks`, joint attention path), AdaLayerNormZeroSingle modulation in single blocks, and `norm_out` before final `proj_out`. | None required for those paths |
| HunyuanImage-3 | `tencent/HunyuanImage3` | All layers | None |
| HunyuanVideo-1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `720p_t2v`, `480p_i2v` | All layers | None |
| Helios | `BestWishYsh/Helios-Base`, `BestWishYsh/Helios-Mid`, `BestWishYsh/Helios-Distilled` | All layers | None |

## Combining with Other Features

FP8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="fp8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```
