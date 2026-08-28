#!/usr/bin/env python3
"""Real end-to-end LoRA training for Cosmos3-Super on 8x Ascend910 (TP=8).

No proxies this time:
  - Real VAE: training images are encoded with the model's own VAE (latent z0).
  - Real text conditioning: the full transformer forward runs the UND language
    model on the tokenized prompt, producing the cross-attention K/V.
  - Full 64-layer GEN pathway + UND backbone, weights streamed from the real
    checkpoint through the pipeline's own remap loader.
  - Flow-matching objective: z_t = (1-t) z0 + t eps, predict velocity eps - z0.
  - LoRA on the column-parallel projections (cross-attn Q/K/V + GEN MLP
    gate/up). A/B kept full-size on every rank; gradients all-reduced so all
    ranks hold identical adapter weights; rank 0 exports the PEFT adapter.

Launch:
  cd /vllm-workspace/vllm-omni
  torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29540 \
      /tmp/train_e2e_lora.py
"""
import torch, torch_npu, json, sys, os, glob, math, random
import numpy as np

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])

STEPS = 40
LORA_R, LORA_ALPHA = 8, 16
BATCH = 1
LR = 1e-3
SZ = 256                      # training image size
DATASET = "/home/ma-user/work/z84450661/lora_datasets/hammershoi/train"
OUT = "/home/ma-user/work/z84450661/lora_datasets/hammershoi/adapter_e2e"
PROMPT = ("hmrsh, an oil painting of an interior room wall, muted palette, "
          "soft window light, danish symbolism style")
SEED = 1234

sys.path.insert(0, "/vllm-workspace/vllm-omni")
from vllm.config import VllmConfig, DeviceConfig
from vllm.config.vllm import set_current_vllm_config

vllm_cfg = VllmConfig()
vllm_cfg.device_config = DeviceConfig(device="npu")

with set_current_vllm_config(vllm_cfg):
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel,
    )
    init_distributed_environment(world_size=WORLD, rank=RANK,
                                 local_rank=LOCAL_RANK,
                                 distributed_init_method="env://", backend="hccl")
    initialize_model_parallel(tensor_parallel_size=WORLD)
    torch.npu.set_device(LOCAL_RANK)

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from safetensors.torch import load_file as load_safetensors

    MODEL = "/run/z84450661/Cosmos3-Super"
    tf_cfg_raw = json.load(open(f"{MODEL}/transformer/config.json"))

    from vllm_omni.diffusion.data import TransformerConfig
    od = OmniDiffusionConfig(model=MODEL, dtype=torch.bfloat16)
    od.tf_model_config = TransformerConfig.from_dict(tf_cfg_raw)

    if RANK == 0:
        print(f"[0] building pipeline (TP={WORLD})")
    pipeline = Cosmos3OmniDiffusersPipeline(od_config=od)
    transformer = pipeline.transformer
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer

    # ---- stream real checkpoint weights through the pipeline's remap loader ----
    if RANK == 0:
        print("[1] loading checkpoint weights")
    idx = json.load(open(f"{MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"))
    wm = idx["weight_map"]

    def stream_weights():
        shard_paths = sorted(set(os.path.join(f"{MODEL}/transformer", t) for t in wm.values()))
        for sp in shard_paths:
            shard = load_safetensors(sp)
            for ck, t in shard.items():
                yield f"transformer.{ck}", t
            del shard

    loaded = pipeline.load_weights(stream_weights())
    if RANK == 0:
        print(f"    loaded {len(loaded)} tensors")

    # move the whole pipeline to this rank's NPU (serving does this in the runner)
    transformer.to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)
    vae.to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)

    # ---- LoRA injection on column-parallel projections ----
    if RANK == 0:
        print("[2] injecting LoRA")
    TP_RANK = torch.distributed.get_rank() % WORLD

    class TPColLoRA(torch.nn.Module):
        """LoRA wrapper for ColumnParallelLinear: full-size A/B on every rank,
        only the local output chunk contributes to the forward."""
        def __init__(self, base, r=LORA_R, alpha=LORA_ALPHA):
            super().__init__()
            self.base = base
            out_local, in_f = base.weight.shape
            self.out_local = out_local
            self.lora_A = torch.nn.Linear(in_f, r, bias=False)
            self.lora_B = torch.nn.Linear(r, out_local * WORLD, bias=False)
            torch.nn.init.normal_(self.lora_A.weight, std=0.02)
            torch.nn.init.zeros_(self.lora_B.weight)
            self.scale = alpha / r
        def forward(self, x):
            y = self.base(x)  # [..., out_local]
            lora_out = self.lora_B(self.lora_A(x))          # [..., out_full]
            chunks = torch.chunk(lora_out, WORLD, dim=-1)
            return y + self.scale * chunks[TP_RANK]

    injected = []
    for name, mod in transformer.named_modules():
        base_name = name.split(".")[-1]
        if base_name in ("to_q", "to_k", "to_v", "gate_proj", "up_proj"):
            parent = transformer
            path = name.split(".")
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], TPColLoRA(mod).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16))
            injected.append(name)
    if RANK == 0:
        print(f"    injected LoRA into {len(injected)} column-parallel modules")
        print(f"    sample: {injected[:3]}")

    lora_params = [p for n, p in transformer.named_parameters() if "lora_" in n]
    if RANK == 0:
        print(f"    LoRA params: {sum(p.numel() for p in lora_params):,}")

    # ---- dataset: real VAE latents ----
    if RANK == 0:
        print("[3] VAE-encoding dataset")
    from PIL import Image
    files = sorted(glob.glob(os.path.join(DATASET, "*.png")))[:32]
    frames = []
    for f in files:
        img = np.array(Image.open(f).convert("RGB").resize((SZ, SZ))).astype(np.float32) / 127.5 - 1.0
        frames.append(torch.from_numpy(img).permute(2, 0, 1))
    video = torch.stack(frames).unsqueeze(2).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)  # [N,3,1,H,W]
    with torch.no_grad():
        z0 = vae.encode(video).latent_dist.mode().to(torch.bfloat16)   # [N, 48, 1, H/16, W/16]
    if RANK == 0:
        print(f"    z0 shape {tuple(z0.shape)} from {len(files)} images")

    # ---- tokenize prompt ----
    tok = tokenizer(PROMPT, return_tensors="pt", padding="max_length",
                    max_length=512, truncation=True)
    text_ids = tok["input_ids"].to(f"npu:{LOCAL_RANK}")
    text_mask = tok["attention_mask"].to(f"npu:{LOCAL_RANK}")
    text_ids = text_ids.repeat(BATCH, 1)
    text_mask = text_mask.repeat(BATCH, 1)
    if RANK == 0:
        print(f"[4] prompt tokens: {tuple(text_ids.shape)} mask {tuple(text_mask.shape)}")

    # ---- train ----
    opt = torch.optim.AdamW(lora_params, lr=LR)
    transformer.train()
    gen = torch.Generator(device=f"npu:{LOCAL_RANK}").manual_seed(SEED)
    print(f"[{RANK}] training {STEPS} steps")
    losses = []
    video_shape = (z0.shape[2], z0.shape[3], z0.shape[4])  # (1, 16, 16)

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        # UND K/V cache must not leak the previous step's autograd graph
        transformer.cached_kv = None
        transformer.cached_freqs_gen = None
        b = random.randint(0, z0.shape[0] - BATCH)
        zb = z0[b:b + BATCH]
        eps = torch.randn_like(zb, generator=gen)
        t = torch.rand(BATCH, device=f"npu:{LOCAL_RANK}", generator=gen, dtype=torch.bfloat16)
        zt = (1 - t.view(-1, 1, 1, 1, 1)) * zb + t.view(-1, 1, 1, 1, 1) * eps
        v_target = eps - zb  # flow-matching velocity

        pred = transformer(hidden_states=zt, timestep=t, text_ids=text_ids,
                           text_mask=text_mask, video_shape=video_shape)
        loss = (pred - v_target).float().pow(2).mean()
        loss.backward()

        # all-reduce LoRA gradients (full-size A/B replicated per rank)
        for p in lora_params:
            if p.grad is not None:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                p.grad /= WORLD
        opt.step()
        losses.append(loss.item())
        if RANK == 0 and (step % 5 == 0 or step == STEPS - 1):
            print(f"step {step:3d}  loss {loss.item():.4f}")

    torch.distributed.barrier()
    if RANK == 0:
        print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ---- save PEFT adapter (all ranks identical after grad all-reduce) ----
    if RANK == 0:
        os.makedirs(OUT, exist_ok=True)
        sd = {}
        for n, p in transformer.named_parameters():
            if "lora_" in n:
                # strip wrapper suffix to get the base module path
                key = "base_model.model." + n
                sd[key] = p.detach().cpu().to(torch.float32)
        from safetensors.torch import save_file
        save_file(sd, os.path.join(OUT, "adapter_model.safetensors"))
        json.dump({
            "peft_type": "LORA", "r": LORA_R, "lora_alpha": LORA_ALPHA,
            "target_modules": ["to_q", "to_k", "to_v", "gate_proj", "up_proj"],
            "lora_dropout": 0.0, "bias": "none", "inference_mode": True,
            "base_model_name_or_path": MODEL, "prompt": PROMPT,
        }, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=2)
        print(f"adapter saved to {OUT} ({len(sd)} tensors)")
    torch.distributed.barrier()
    if RANK == 0:
        print("E2E LoRA TRAINING DONE")
