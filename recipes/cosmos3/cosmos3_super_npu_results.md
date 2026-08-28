# Cosmos3-Super NPU Results (8× Ascend910, TP=8)

All tests run on 8 Ascend910 NPU devices (davinci 8–15, NPU chips 4–7) with
`--tensor-parallel-size 8`, `--no-guardrails`, `bf16`. Official params:
`size=1280x720, num_frames=189, fps=24, num_inference_steps=35,
guidance_scale=6.0, flow_shift=10.0, max_sequence_length=4096`.

| # | Mode | Prompt | Input | Output | Size / Steps | Time |
|---|---|---|---|---|---|---|
| 1 | T2I | "A photorealistic red sports car on a city street at golden hour, cinematic lighting." | -- | `cosmos3_super_t2i.png` (3.1 MB) | 1024×1024 / 50 | -- |
| 2 | T2V | "A robot arm is cleaning a plate in the kitchen" | -- | `cosmos3_super_t2v.mp4` | 1280×720 189f / 35 | 8m18s |
| 3 | I2V | "A car (viewed from dashcam POV) is traveling fast along a coastal mountain road, surrounded by steep rocky cliffs. Suddenly, rocks and debris tumble down from the cliff face onto the road, forcing the car to make an emergency stop. Dust clouds billow as rocks crash around the vehicle." | `cosmos3_super_i2v_input.jpg` (841 KB) | `cosmos3_super_i2v.mp4` | 1280×720 189f / 35 | 10m43s |
| 4 | V2V | "A robotic arm is carefully cleaning a white plate in a bright kitchen. The robot arm moves methodically, wiping the plate surface with a cloth, with kitchen appliances visible in the background." | `cosmos3_super_v2v_input.mp4` (1.7 MB) | `cosmos3_super_v2v.mp4` | 1280×720 189f / 35 | 8m22s |
| 5 | T2VS | "A robot arm is cleaning a plate in the kitchen" | -- | `cosmos3_super_t2vs.mp4` (10.4 MB, AAC 48 kHz stereo) | 1280×720 189f / 35 | 9m21s |
| 6 | I2VS | "The scene comes to life with smooth, natural motion and ambient sound." | `cosmos3_super_i2v_input.jpg` (841 KB) | `cosmos3_super_i2vs.mp4` (12.2 MB, AAC 48 kHz stereo) | 1280×720 189f / 35 | 9m17s |

## Artifacts

All files in `/home/ma-user/work/z84450661/`:

```
cosmos3_super_i2v_input.jpg       841 KB   Reference image (from Cosmos3-Super assets)
cosmos3_super_v2v_input.mp4       1.7 MB   Reference video (example_t2v_diffusers_output.mp4)
cosmos3_super_t2vs.mp4           10.4 MB   T2V + synchronized audio (AAC 48 kHz stereo)
cosmos3_super_i2vs.mp4           12.2 MB   I2V + synchronized audio (AAC 48 kHz stereo)
```

## Server Config

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
vllm serve /run/z84450661/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --init-timeout 1800
```
