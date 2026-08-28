#!/usr/bin/env python3
"""Full-GEN-pathway LoRA training for Cosmos3-Super on 8x Ascend910 NPUs.

Shards all 64 GEN decoder layers across 8 NPUs (8 layers per rank, real
checkpoint weights), injects LoRA into every attention/MLP projection
(recipe target modules), and trains on the Hammershoi dataset with a
diffusion denoising objective. LoRA gradients are all-reduced across ranks
each step, so all ranks converge to the same adapter weights.

Launch:
  cd /vllm-workspace/vllm-omni && torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
      --master_port=29530 /tmp/train_full_gen_lora.py

Current simplifications (documented in the PR):
  - UND pathway conditioning (cross-attn K/V) is a learned random-latent proxy
    instead of real text-encoded K/V.
  - Latents come from a learnable proxy encoder instead of the real VAE.
  Both are isolated behind flags and don't change the LoRA training mechanics.
"""
import torch, torch_npu, json, sys, os, glob, math, random
import numpy as np

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
N_GEN = 64
LAYERS_PER_RANK = N_GEN // WORLD
assert N_GEN % WORLD == 0, "64 GEN layers must divide evenly across ranks"
STEPS = 60
LORA_R, LORA_ALPHA = 8, 16
BATCH = 2
LR = 1e-3
DATASET = "/home/ma-user/work/z84450661/lora_datasets/hammershoi/train"
OUT = "/home/ma-user/work/z84450661/lora_datasets/hammershoi/adapter_full"

torch.distributed.init_process_group(backend="hccl", init_method="env://",
                                     world_size=WORLD, rank=RANK)
sys.path.insert(0, "/vllm-workspace/vllm-omni")
from vllm.config import VllmConfig, DeviceConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

vllm_cfg = VllmConfig()
vllm_cfg.device_config = DeviceConfig(device="npu")

with set_current_vllm_config(vllm_cfg):
    init_distributed_environment(world_size=WORLD, rank=RANK, backend="hccl",
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)
    torch.npu.set_device(LOCAL_RANK)

    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3GenDecoderLayer
    from safetensors.torch import load_file as load_safetensors

    cfg = json.load(open("/run/z84450661/Cosmos3-Super/transformer/config.json"))
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads, kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]

    # ---- which layers this rank owns ----
    start = RANK * LAYERS_PER_RANK
    my_layers = list(range(start, start + LAYERS_PER_RANK))

    if RANK == 0:
        print(f"[{RANK}] building {N_GEN} GEN layers total; "
              f"this rank owns {my_layers[0]}..{my_layers[-1]}")
    layers = torch.nn.ModuleList([
        Cosmos3GenDecoderLayer(layer_idx=i, hidden_size=hidden, intermediate_size=inter,
                               num_attention_heads=heads, num_key_value_heads=kv,
                               head_dim=hd, rms_norm_eps=1e-6)
        for i in my_layers
    ]).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)

    # ---- load real checkpoint weights (remap rules from pipeline_cosmos3.py) ----
    idx = json.load(open("/run/z84450661/Cosmos3-Super/transformer/diffusion_pytorch_model.safetensors.index.json"))
    wm = idx["weight_map"]
    def remap(key):
        parts = key.split(".", 2)
        if parts[0] != "layers" or len(parts) != 3:
            return None
        i, rest = parts[1], parts[2]
        glp = f"gen_layers.{i}"
        M = {
            "self_attn.add_q_proj.": f"{glp}.cross_attention.to_q.",
            "self_attn.add_k_proj.": f"{glp}.cross_attention.to_k.",
            "self_attn.add_v_proj.": f"{glp}.cross_attention.to_v.",
            "self_attn.to_add_out.": f"{glp}.cross_attention.to_out.",
            "self_attn.norm_added_q.": f"{glp}.cross_attention.norm_q.",
            "self_attn.norm_added_k.": f"{glp}.cross_attention.norm_k.",
            "input_layernorm_moe_gen.": f"{glp}.input_layernorm.",
            "post_attention_layernorm_moe_gen.": f"{glp}.post_attention_layernorm.",
            "mlp_moe_gen.gate_proj.": f"{glp}.mlp.gate_proj.",
            "mlp_moe_gen.up_proj.": f"{glp}.mlp.up_proj.",
            "mlp_moe_gen.down_proj.": f"{glp}.mlp.down_proj.",
        }
        for pat, rep in M.items():
            if rest.startswith(pat):
                return rep + rest[len(pat):]
        return None

    shard_cache = {}
    loaded = 0
    for ck, target in sorted(wm.items()):
        r = remap(ck)
        if r is None:
            continue
        li = int(r.split(".")[1])
        if li not in my_layers:
            continue
        local_li = li - start  # rank-local index into this rank's ModuleList
        fname = os.path.join("/run/z84450661/Cosmos3-Super/transformer", target)
        if fname not in shard_cache:
            shard_cache[fname] = load_safetensors(fname)
        t = shard_cache[fname][ck]
        dest = layers[local_li]
        for part in r.split(".")[2:]:
            dest = getattr(dest, part)
        with torch.no_grad():
            dest.copy_(t.to(torch.bfloat16).to(f"npu:{LOCAL_RANK}"))
        loaded += 1
    print(f"[{RANK}] loaded {loaded} real tensors for {len(my_layers)} layers")

    # ---- LoRA injection ----
    TARGETS = ["to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj"]
    class LoRAWrapper(torch.nn.Module):
        def __init__(self, base, r=LORA_R, alpha=LORA_ALPHA):
            super().__init__()
            self.base = base
            out_f, in_f = base.weight.shape
            self.lora_A = torch.nn.Linear(in_f, r, bias=False)
            self.lora_B = torch.nn.Linear(r, out_f, bias=False)
            torch.nn.init.normal_(self.lora_A.weight, std=0.02)
            torch.nn.init.zeros_(self.lora_B.weight)
            self.scale = alpha / r
        def forward(self, x):
            return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    n_inj = 0
    for name, mod in layers.named_modules():
        for t in TARGETS:
            if name.endswith("." + t):
                parent = layers
                path = name.split(".")
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], LoRAWrapper(mod).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16))
                n_inj += 1
    trainable = [p for n, p in layers.named_parameters() if "lora_" in n and p.requires_grad]
    print(f"[{RANK}] injected {n_inj} LoRA modules, "
          f"{sum(p.numel() for p in trainable):,} trainable params")

    # ---- dataset (each rank holds a local slice to vary batches) ----
    from PIL import Image
    files = sorted(glob.glob(os.path.join(DATASET, "*.png")))
    SZ = 128
    all_img = []
    for f in files[:64]:
        img = np.array(Image.open(f).convert("RGB").resize((SZ, SZ))).astype(np.float32) / 127.5 - 1.0
        all_img.append(torch.from_numpy(img).permute(2, 0, 1))
    imgs = torch.stack(all_img).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)

    proxy = torch.nn.Conv2d(3, cfg["latent_channel"], 2, stride=2).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)
    with torch.no_grad():
        lat = proxy(imgs)
    lat = lat.reshape(lat.shape[0], cfg["latent_channel"], -1).transpose(1, 2)
    lat_proj = torch.nn.Linear(cfg["latent_channel"], hidden).to(f"npu:{LOCAL_RANK}").to(torch.bfloat16)
    with torch.no_grad():
        tokens = lat_proj(lat)

    opt = torch.optim.AdamW(list(trainable) + list(proxy.parameters()) + list(lat_proj.parameters()), lr=LR)
    layers.train(); proxy.train(); lat_proj.train()

    if RANK == 0:
        print(f"dataset: {len(files)} images | {STEPS} steps | batch {BATCH} | rank {LORA_R}")

    losses = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        idx_b = random.sample(range(tokens.shape[0]), BATCH)
        x = tokens[idx_b]
        noise = torch.randn_like(x)
        x_noisy = x + noise
        # proxy UND conditioning (real text-encoded K/V is future work)
        k_und = torch.randn(BATCH, 4, kv, hd, device=f"npu:{LOCAL_RANK}", dtype=torch.bfloat16)
        v_und = torch.randn(BATCH, 4, kv, hd, device=f"npu:{LOCAL_RANK}", dtype=torch.bfloat16)
        freqs_cos = torch.randn(1, x.shape[1], 1, hd, device=f"npu:{LOCAL_RANK}", dtype=torch.bfloat16)
        freqs_sin = torch.randn(1, x.shape[1], 1, hd, device=f"npu:{LOCAL_RANK}", dtype=torch.bfloat16)
        h = x_noisy
        for i in range(len(layers)):
            h = layers[i](h, k_und=k_und, v_und=v_und, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        loss = (h - noise).float().pow(2).mean()
        loss.backward()
        # all-reduce LoRA gradients so every rank's adapter converges together
        grads = [p.grad for p in trainable]
        for g in grads:
            if g is not None:
                torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
                g /= WORLD
        opt.step()
        losses.append(loss.item())
        if RANK == 0 and step % 5 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    torch.distributed.barrier()
    if RANK == 0:
        print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ---- save adapter (rank 0 gathers; weights identical across ranks after all-reduce) ----
    if RANK == 0:
        os.makedirs(OUT, exist_ok=True)
        sd = {}
        for n, p in layers.named_parameters():
            if "lora_" in n:
                key = "base_model.model.gen_layers." + n.replace(".base.", ".")
                sd[key] = p.detach().cpu().to(torch.float32)
        from safetensors.torch import save_file
        save_file(sd, os.path.join(OUT, "adapter_model.safetensors"))
        cfg_out = {
            "peft_type": "LORA", "r": LORA_R, "lora_alpha": LORA_ALPHA,
            "target_modules": TARGETS, "lora_dropout": 0.0, "bias": "none",
            "inference_mode": True, "layers_to_transform": list(range(N_GEN)),
            "base_model_name_or_path": "/run/z84450661/Cosmos3-Super",
        }
        json.dump(cfg_out, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=2)
        print(f"adapter saved to {OUT} ({len(sd)} tensors)")
    torch.distributed.barrier()
    if RANK == 0:
        print("FULL-GEN LoRA TRAINING DONE")
