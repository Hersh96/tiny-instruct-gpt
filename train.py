"""Training loop for both stages.

  python train.py --stage pretrain --preset small
  python train.py --stage sft --init_from out/pretrain/ckpt.pt

Features: bf16/fp16 autocast, torch.compile, gradient accumulation, cosine LR
with warmup, gradient clipping, periodic eval, CSV logging, checkpointing.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import time

import torch

from tinygpt.data import PretrainLoader, SFTLoader
from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import BPETokenizer

PRESETS = {
    # ~1.4M params: CPU smoke tests
    "tiny": dict(n_layer=2, n_head=4, n_kv_head=2, n_embd=128, block_size=256),
    # ~13M params (9M non-embedding): free Colab T4 in about an hour
    "small": dict(n_layer=6, n_head=6, n_kv_head=2, n_embd=384, block_size=512),
    # ~42M params (36M non-embedding): one 16-24GB GPU in a few hours
    "base": dict(n_layer=8, n_head=8, n_kv_head=4, n_embd=640, block_size=512),
}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--preset", choices=list(PRESETS), default="small")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--init_from", default=None, help="checkpoint to start from (required for sft)")
    # model overrides / ablations
    ap.add_argument("--n_layer", type=int)
    ap.add_argument("--n_head", type=int)
    ap.add_argument("--n_kv_head", type=int)
    ap.add_argument("--n_embd", type=int)
    ap.add_argument("--block_size", type=int)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--norm", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    ap.add_argument("--mlp", choices=["swiglu", "gelu"], default="swiglu")
    ap.add_argument("--pos", choices=["rope", "learned"], default="rope")
    # optimisation
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_iters", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # runtime
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # stage-specific defaults
    sft = args.stage == "sft"
    args.max_iters = args.max_iters or (1500 if sft else 5000)
    args.lr = args.lr or (1e-4 if sft else 1e-3)
    args.warmup_iters = args.warmup_iters if args.warmup_iters is not None else (50 if sft else 200)
    args.out_dir = args.out_dir or os.path.join("out", args.stage)
    if args.dtype is None:
        if args.device.startswith("cuda"):
            args.dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            args.dtype = "float32"
    if args.compile is None:
        args.compile = args.device.startswith("cuda")
    if sft and not args.init_from:
        ap.error("--stage sft needs --init_from <pretrain checkpoint>")
    return args


def lr_at(it: int, args) -> float:
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = BPETokenizer.load(os.path.join(args.data_dir, "tokenizer.json"))

    # ---- model
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ckpt["config"])
        cfg.dropout = args.dropout
        model = GPT(cfg)
        model.load_state_dict(ckpt["model"])
        print(f"initialised from {args.init_from} (iter {ckpt.get('iter')})")
    else:
        overrides = {k: getattr(args, k) for k in ("n_layer", "n_head", "n_kv_head", "n_embd", "block_size") if getattr(args, k)}
        cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            dropout=args.dropout,
            norm=args.norm,
            mlp=args.mlp,
            pos=args.pos,
            **{**PRESETS[args.preset], **overrides},
        )
        model = GPT(cfg)
    model.to(args.device)
    print(f"model: {model.num_params() / 1e6:.2f}M non-embedding params, {model.num_params(False) / 1e6:.2f}M total")
    print(f"config: {cfg.to_dict()}")

    # ---- data
    if args.stage == "pretrain":
        train_dl = PretrainLoader(os.path.join(args.data_dir, "pretrain_train.bin"), args.batch_size, cfg.block_size, args.device)
        val_dl = PretrainLoader(os.path.join(args.data_dir, "pretrain_val.bin"), args.batch_size, cfg.block_size, args.device)
    else:
        pad = tok.special["<|pad|>"]
        train_dl = SFTLoader(os.path.join(args.data_dir, "sft_train.npz"), args.batch_size, cfg.block_size, args.device, pad)
        val_dl = SFTLoader(os.path.join(args.data_dir, "sft_val.npz"), args.batch_size, cfg.block_size, args.device, pad)

    optimizer = model.configure_optimizer(args.lr, args.weight_decay, device_type=device_type)
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx = torch.autocast(device_type, dtype=ptdtype) if ptdtype != torch.float32 else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16"))

    train_model = torch.compile(model) if args.compile else model
    if args.compile:
        print("compiling model with torch.compile ...")

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split, dl in (("train", train_dl), ("val", val_dl)):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y, m = dl.batch()
                with ctx:
                    _, loss = train_model(x, y, loss_mask=m)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    log_path = os.path.join(args.out_dir, "log.csv")
    log_f = open(log_path, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "tokens", "train_loss", "val_loss", "lr", "tok_per_sec"])

    tokens_per_iter = args.batch_size * args.grad_accum * cfg.block_size
    print(f"tokens per iteration (upper bound for sft): {tokens_per_iter:,}")
    best_val, tokens_seen = float("inf"), 0
    x, y, m = train_dl.batch()
    t0 = time.time()
    tok_window, t_window = 0, time.time()

    for it in range(args.max_iters + 1):
        lr = lr_at(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = evaluate()
            print(f"[eval] iter {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            log.writerow([it, tokens_seen, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}", ""])
            log_f.flush()
            if losses["val"] < best_val and it > 0:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(), "iter": it, "val_loss": best_val,
                     "stage": args.stage, "args": vars(args)},
                    os.path.join(args.out_dir, "ckpt.pt"),
                )
                print(f"  saved checkpoint (val {best_val:.4f})")
        if it == args.max_iters:
            break

        loss_acc = 0.0
        for _ in range(args.grad_accum):
            with ctx:
                _, loss = train_model(x, y, loss_mask=m)
                loss = loss / args.grad_accum
            ntok = x.numel() if m is None else int(m.sum())
            tokens_seen += ntok
            tok_window += x.numel()
            x, y, m = train_dl.batch()  # prefetch next batch while GPU works
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        if args.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % args.log_interval == 0:
            dt = time.time() - t_window
            tps = tok_window / dt if dt > 0 else 0
            print(f"iter {it}: loss {loss_acc:.4f}, lr {lr:.2e}, {tps:,.0f} tok/s, elapsed {time.time() - t0:.0f}s")
            log.writerow([it, tokens_seen, f"{loss_acc:.4f}", "", f"{lr:.2e}", f"{tps:.0f}"])
            tok_window, t_window = 0, time.time()

    log_f.close()
    print(f"done. best val loss {best_val:.4f}. log: {log_path}")


if __name__ == "__main__":
    main()
