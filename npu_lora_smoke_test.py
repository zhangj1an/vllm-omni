#!/usr/bin/env python3
"""NPU LoRA training smoke test: real Cosmos3-Super GEN layer, manual LoRA injection,
full train loop (fwd/bwd/AdamW) on NPU.

peft 不支持 vllm 的 ColumnParallelLinear 注入(serve 侧由 DiffusionLoRAManager 处理);
训练侧用等价的低秩分解验证 NPU 训练机制。
"""
import torch, torch_npu, json, sys, os
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
sys.path.insert(0, "/vllm-workspace/vllm-omni")

torch.distributed.init_process_group(backend="hccl", init_method="env://",
                                     world_size=1, rank=0)
from vllm.config import VllmConfig, DeviceConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

vllm_cfg = VllmConfig()
vllm_cfg.device_config = DeviceConfig(device="npu")

with set_current_vllm_config(vllm_cfg):
    init_distributed_environment(world_size=1, rank=0, backend="hccl",
                                 distributed_init_method="env://")
    initialize_model_parallel(tensor_model_parallel_size=1)
    torch.npu.set_device(0)

    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3GenDecoderLayer

    cfg = json.load(open("/run/z84450661/Cosmos3-Super/transformer/config.json"))
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads, kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    print(f"layer config: hidden={hidden} inter={inter} heads={heads}/{kv} head_dim={hd}")

    layer = Cosmos3GenDecoderLayer(
        layer_idx=0, hidden_size=hidden, intermediate_size=inter,
        num_attention_heads=heads, num_key_value_heads=kv, head_dim=hd,
        rms_norm_eps=1e-6,
    ).to("npu:0").to(torch.bfloat16)

    # --- manual LoRA injection (low-rank update on real layer projections) ---
    TARGETS = ["to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj"]
    class LoRAWrapper(torch.nn.Module):
        def __init__(self, base, r=8, alpha=16):
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

    n_injected = 0
    for name, mod in layer.named_modules():
        for t in TARGETS:
            if name.endswith("." + t):
                parent = layer
                path = name.split(".")
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], LoRAWrapper(mod).to("npu:0").to(torch.bfloat16))
                n_injected += 1
    print(f"LoRA injected into {n_injected} projection modules")

    trainable = [p for n, p in layer.named_parameters() if "lora_" in n and p.requires_grad]
    print(f"LoRA trainable params: {sum(p.numel() for p in trainable):,}")

    B, T, C = 1, 8, hidden
    x = torch.randn(B, T, C, device="npu:0", dtype=torch.bfloat16, requires_grad=True)
    k_und = torch.randn(B, 4, kv, hd, device="npu:0", dtype=torch.bfloat16)
    v_und = torch.randn(B, 4, kv, hd, device="npu:0", dtype=torch.bfloat16)
    freqs_cos = torch.randn(1, T, 1, hd, device="npu:0", dtype=torch.bfloat16)
    freqs_sin = torch.randn(1, T, 1, hd, device="npu:0", dtype=torch.bfloat16)

    opt = torch.optim.AdamW(trainable, lr=1e-3)
    layer.train()
    print("\nstep  |  loss")
    losses = []
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        out = layer(x, k_und=k_und, v_und=v_und, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        g = sum(p.grad.abs().sum().item() for p in trainable if p.grad is not None)
        print(f"{step:4d}  |  {loss.item():.6f}  grad_sum={g:.3e}")

    print(f"\nfirst LoRA-B weight: {trainable[1].flatten()[:3].tolist()}")
    assert losses[-1] < losses[0] * 0.98, "loss did not decrease"
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    print(f"\nLoRA gradients flowing: {grad_ok}")
    print(f"loss {losses[0]:.6f} -> {losses[-1]:.6f}  ({losses[-1]/losses[0]*100:.1f}% of initial)")
    print("NPU LoRA TRAINING PATH: OK")
