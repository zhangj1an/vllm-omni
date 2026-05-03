"""Monkeypatch torch.nn.functional.scaled_dot_product_attention to log every
call's (B, H, S, D, mask) at runtime, so we can answer the colleague's bullet
#2 ("real shapes per layer") empirically rather than analytically.

Activate by setting `SDPA_SHAPE_LOG=path/to/log.jsonl` in env and running any
diffusion driver. Each call appends one JSON line:

  {"q_shape":[B,H,S,D],"k_shape":[B,H,S,D],"v_shape":[B,H,S,D],
   "mask_shape":[...] or null, "is_causal":bool, "scale":float}

After the run, summarise with:
  python bench_out/sdpa_shape_logger.py --summarize path/to/log.jsonl
"""
import json
import os
import sys


def _install_logger():
    log_path = os.environ.get("SDPA_SHAPE_LOG")
    if not log_path:
        return
    import torch  # noqa: PLC0415

    orig = torch.nn.functional.scaled_dot_product_attention
    log_file = open(log_path, "w", buffering=1)
    counter = {"n": 0}

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        counter["n"] += 1
        try:
            entry = {
                "i": counter["n"],
                "q_shape": list(query.shape),
                "k_shape": list(key.shape),
                "v_shape": list(value.shape),
                "mask_shape": list(attn_mask.shape) if attn_mask is not None else None,
                "mask_dtype": str(attn_mask.dtype) if attn_mask is not None else None,
                "is_causal": bool(is_causal),
                "scale": float(scale) if scale is not None else None,
                "dtype": str(query.dtype),
            }
            log_file.write(json.dumps(entry) + "\n")
        except Exception as e:
            log_file.write(json.dumps({"i": counter["n"], "log_error": str(e)}) + "\n")
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)

    torch.nn.functional.scaled_dot_product_attention = wrapped
    sys.stderr.write(f"[sdpa_shape_logger] installed; logging to {log_path}\n")


def _summarize(path: str) -> None:
    from collections import Counter

    shapes = Counter()
    masks = Counter()
    n = 0
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "log_error" in e:
                continue
            n += 1
            q = tuple(e["q_shape"])
            k = tuple(e["k_shape"])
            shapes[(q, k)] += 1
            mask_kind = "none" if e["mask_shape"] is None else f"mask{tuple(e['mask_shape'])}"
            masks[mask_kind] += 1
    print(f"Total SDPA calls: {n}")
    print()
    print("Distinct (Q, K) shapes (top 20 by frequency):")
    print(f"{'count':>8}  {'Q shape':<30}  {'K shape':<30}")
    print("-" * 75)
    for (q, k), c in shapes.most_common(20):
        print(f"{c:>8}  {str(q):<30}  {str(k):<30}")
    print()
    print("Mask shapes:")
    for m, c in masks.most_common():
        print(f"{c:>8}  {m}")


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--summarize":
    _summarize(sys.argv[2])
else:
    _install_logger()
