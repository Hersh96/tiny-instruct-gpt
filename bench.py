"""Speed benchmarks: what each optimisation buys you.

  python bench.py --preset small            # training + generation tables
  python bench.py --preset small --skip_train

Uses random weights and random tokens, so no data or checkpoint is needed.
Writes a Markdown table to assets/benchmarks.md.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch

from tinygpt.model import GPT, GPTConfig
from train import PRESETS


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_train(cfg, device, dtype, compile_, batch, steps=20, warmup=5):
    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(1e-4, 0.1, device_type="cuda" if device.startswith("cuda") else "cpu")
    m = torch.compile(model) if compile_ else model
    ctx = torch.autocast(device.split(":")[0], dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=dtype == torch.float16)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    for i in range(warmup + steps):
        if i == warmup:
            sync(device)
            t0 = time.perf_counter()
        with ctx:
            _, loss = m(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    sync(device)
    return steps * batch * cfg.block_size / (time.perf_counter() - t0)


@torch.no_grad()
def bench_generate(cfg, device, use_cache, prompt_len=16, new_tokens=256, dtype=torch.float32):
    torch.manual_seed(0)
    model = GPT(cfg).to(device=device, dtype=dtype).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    for _ in model.generate(idx, 8, temperature=1.0, use_cache=use_cache):
        pass  # warmup
    sync(device)
    t0 = time.perf_counter()
    n = 0
    for _ in model.generate(idx, new_tokens, temperature=1.0, use_cache=use_cache):
        n += 1
    sync(device)
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--out", default="assets/benchmarks.md")
    args = ap.parse_args()

    cfg = GPTConfig(vocab_size=8192, **PRESETS[args.preset])
    hw = torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU"
    lines = [f"## Benchmarks: preset `{args.preset}` on {hw}\n"]

    if not args.skip_train:
        rows = [("fp32, eager", torch.float32, False)]
        if args.device.startswith("cuda"):
            half = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            name = "bf16" if half == torch.bfloat16 else "fp16"
            rows += [(f"{name} autocast, eager", half, False), (f"{name} autocast + torch.compile", half, True)]
        lines += ["### Training throughput\n", "| setup | tokens/sec | speedup |", "|---|---:|---:|"]
        base = None
        for name, dt, comp in rows:
            tps = bench_train(cfg, args.device, dt, comp, args.batch)
            base = base or tps
            lines.append(f"| {name} | {tps:,.0f} | {tps / base:.2f}x |")
            print(lines[-1])
        lines.append("")

    lines += ["### Generation speed (batch 1, 256 new tokens)\n", "| setup | tokens/sec | speedup |", "|---|---:|---:|"]
    no_cache = bench_generate(cfg, args.device, use_cache=False)
    cache = bench_generate(cfg, args.device, use_cache=True)
    lines.append(f"| no KV-cache (recompute full context) | {no_cache:,.0f} | 1.00x |")
    lines.append(f"| KV-cache | {cache:,.0f} | {cache / no_cache:.2f}x |")
    print("\n".join(lines[-2:]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
