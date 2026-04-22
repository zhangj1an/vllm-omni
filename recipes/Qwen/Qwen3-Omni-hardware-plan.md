# Qwen3-Omni (Planning Scaffold)

> **Status:** Planning document for
> [issue #2645](https://github.com/vllm-project/vllm-omni/issues/2645).
> Structure follows
> [`recipes/Wan-AI/Wan2.2-I2V.md`](../Wan-AI/Wan2.2-I2V.md) with additions
> borrowed from
> [vllm-project/recipes → DeepSeek/DeepSeek-V3.md](https://github.com/vllm-project/recipes/blob/main/DeepSeek/DeepSeek-V3.md)
> (`## Running …` intro + `<details>` parallelism variants + `## Benchmarking`).
> TODO / TBD markers flag the blanks that need real-hardware validation before
> this lands as a recipe.

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Task: Multimodal chat with text, image, audio, or video input (3-stage
  pipeline: thinker → talker → code2wav)
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Qwen/Qwen3-Omni-30B-A3B-Instruct` with vLLM-Omni on one of the supported
hardware platforms and validating the deployment with the bundled multimodal
client examples.

## References

- Upstream model card: <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>
- Related docs:
  [`docs/user_guide/examples/online_serving/qwen3_omni.md`](../../docs/user_guide/examples/online_serving/qwen3_omni.md)
- Related example:
  [`examples/online_serving/qwen3_omni/README.md`](../../examples/online_serving/qwen3_omni/README.md)
- Stage layouts (authoritative):
  [`vllm_omni/deploy/qwen3_omni_moe.yaml`](../../vllm_omni/deploy/qwen3_omni_moe.yaml)
- RFC: [issue #2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Running Qwen3-Omni

Qwen3-Omni-30B-A3B-Instruct is a 30B MoE (~3B active) with a 3-stage
generation pipeline; BF16 weights are ~60 GB. The following launch profiles
are planned — see the per-platform subsections below for commands.

- **1x A100 80GB (BF16)** — single-card reference, already documented
- **2x H100 80GB (BF16)** — verified stage layout in `deploy/qwen3_omni_moe.yaml`
- **2x L40S 48GB (BF16)** — same layout, TBD
- **4–8x RTX 4090 24GB (quantized)** — feasibility TBD
- **1–2x AMD MI300X (ROCm, BF16)** — TBD
- **5x Intel Arc Pro B-series (XPU, BF16)** — configured in deploy YAML, TBD
- **3x Huawei Ascend (NPU, BF16)** — configured in deploy YAML, TBD

## Hardware Support

## CUDA GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA with one A100 80 GB GPU
- vLLM / vLLM-Omni: match repository requirements for your checkout

#### Command

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

Async-chunk variant:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/deploy/qwen3_omni_moe.yaml
```

#### Verification

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --query-type use_image \
  --port 8091 \
  --host localhost
```

#### Notes

- `--omni` is required.
- Current starter section in [`Qwen3-Omni.md`](./Qwen3-Omni.md) documents
  this config; this plan extends coverage to additional hardware.

### 2x H100 80GB

#### Environment

- OS: Linux
- Driver / runtime: NVIDIA CUDA with 2x H100 80 GB GPUs
- Stage layout (from `deploy/qwen3_omni_moe.yaml`): stage 0 on `cuda:0`,
  stages 1+2 co-located on `cuda:1`

#### Command

<details>
<summary>Staged layout (thinker on GPU0, talker + code2wav on GPU1) — <b>verified layout, exact command TBD</b></summary>

```bash
# TODO: confirm deploy.yaml path/flag and any env-var requirements
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/deploy/qwen3_omni_moe.yaml
```

</details>

#### Verification

TODO: reuse the multimodal client example from the A100 section.

#### Notes

- TODO: pin `gpu_memory_utilization` tuning advice and any cudagraph caveats.

### 2x L40S 48GB

> **Status:** TBD — needs validation.

#### Environment

- NVIDIA CUDA with 2x L40S (48 GB each)
- Expected to reuse the H100 staged layout; watch stage 0 headroom

#### Command

<details>
<summary>Staged layout — <b>TBD</b></summary>

```bash
# TODO: validate memory headroom and adjust gpu_memory_utilization for stage 0
```

</details>

#### Verification / Notes

TODO.

### 4–8x RTX 4090 24GB

> **Status:** TBD — feasibility unknown, likely requires a quantized
> checkpoint. 30B BF16 ≈ 60 GB of weights; 4090 is 24 GB per card.

#### Environment

- NVIDIA CUDA with 4–8x RTX 4090 (24 GB each)
- TODO: determine whether an AWQ / GPTQ checkpoint exists or can be produced

#### Command

<details>
<summary>TP ≥ 4 on stage 0 (quantized) — <b>TBD</b></summary>

```bash
# TODO: command pending quantization strategy
```

</details>

#### Verification / Notes

TODO.

## AMD GPU (ROCm)

### 1–2x MI300X

> **Status:** TBD. `deploy/qwen3_omni_moe.yaml` has a `rocm:` override
> (`enforce_eager: true` on stage 0) but otherwise keeps the 2-device layout.
> On a single MI300X (192 GB) the layout may collapse to one card.

#### Environment

- ROCm version: TODO (pin)
- Python: 3.10+
- GPU: MI300X (or MI325X / MI355X as community validates)

#### Prerequisites

TODO: ROCm-specific install steps, `miopen` cache directory if needed.

#### Command

<details>
<summary>BF16 — <b>TBD</b></summary>

```bash
# TODO
```

</details>

<details>
<summary>FP8 — <b>TBD, confirm FP8 support path for Qwen3-Omni on ROCm</b></summary>

```bash
# TODO
```

</details>

#### Verification / Notes

TODO.

## Intel XPU

### 5x Arc Pro B-series

> **Status:** TBD. Stage layout from
> `deploy/qwen3_omni_moe.yaml → platforms.xpu`:
> - Stage 0 with TP=4 on devices 0–3, `enforce_eager: true`,
>   `max_cudagraph_capture_size: 0`
> - Stage 1 on device 4, `enforce_eager: true`
> - Stage 2 on device 4, `gpu_memory_utilization: 0.3`

#### Environment

- Intel XPU runtime: TODO (pin oneAPI / driver versions)
- Cards: 5x Arc Pro B50 (16 GB) / B60 (24 GB) / B70 (32 GB)

#### Prerequisites

TODO.

#### Command

<details>
<summary>TP=4 on stage 0, stages 1+2 co-located on card 4 — <b>TBD</b></summary>

```bash
# TODO
```

</details>

#### Verification / Notes

TODO.

## Huawei NPU

### 3x Ascend (A2 / A3)

> **Status:** TBD. Stage layout from
> `deploy/qwen3_omni_moe.yaml → platforms.npu`:
> - Stage 0 with TP=2 on devices 0–1, `gpu_memory_utilization: 0.6`
> - Stage 1 on device 2, `enforce_eager: true`
> - Stage 2 on device 2, `gpu_memory_utilization: 0.3`

#### Environment

- Driver / runtime: Ascend NPU driver + CANN toolkit (pin version)
- Python: 3.10+

#### Prerequisites

TODO: whether **mindie-sd** or similar Ascend operator libraries need to be
installed (see the Wan2.2-I2V recipe for precedent).

#### Command

<details>
<summary>TP=2 on stage 0, stages 1+2 co-located on NPU 2 — <b>TBD</b></summary>

```bash
# TODO
```

</details>

#### Verification / Notes

TODO.

## Benchmarking

TODO: wire this section up against `benchmarks/qwen3-omni/`.

### Benchmark Configurations

- TODO: workload shapes (prompt / output / modality mix)
- TODO: batch sizes / concurrency levels

### Benchmark Command

```bash
# TODO: vllm bench serve invocation pointing at the server launched above
```

### Expected Output

```
TODO: sample benchmark output block
```

## Open Questions

- Which subset of platforms will be first-party? Issue #2645 explicitly allows
  additional hardware sections to land via community validation.
- RTX 4090: confirm whether a quantized variant is worth documenting, or skip.
- 2x L40S: validate before documenting.
- ROCm FP8: confirm support path for Qwen3-Omni on MI300X.
