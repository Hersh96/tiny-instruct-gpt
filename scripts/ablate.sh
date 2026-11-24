#!/usr/bin/env bash
# Ablation: start from a GPT-2 style block and turn on one modern component at a time.
# Each run uses the same data, seed, steps and parameter budget (SwiGLU is sized to match GELU).
set -euo pipefail
ITERS=${ITERS:-2000}
PRESET=${PRESET:-small}
COMMON="--stage pretrain --preset $PRESET --max_iters $ITERS --eval_interval 200"

python train.py $COMMON --out_dir out/ablate_gpt2      --norm layernorm --mlp gelu   --pos learned --n_kv_head 6
python train.py $COMMON --out_dir out/ablate_+rmsnorm  --norm rmsnorm   --mlp gelu   --pos learned --n_kv_head 6
python train.py $COMMON --out_dir out/ablate_+rope     --norm rmsnorm   --mlp gelu   --pos rope    --n_kv_head 6
python train.py $COMMON --out_dir out/ablate_+swiglu   --norm rmsnorm   --mlp swiglu --pos rope    --n_kv_head 6
python train.py $COMMON --out_dir out/ablate_+gqa      --norm rmsnorm   --mlp swiglu --pos rope    --n_kv_head 2

python scripts/plot.py out/ablate_*/log.csv --val_only --title "Ablation: validation loss" --out assets/ablations.png
python - <<'EOF'
import csv, glob, os
rows = []
for p in sorted(glob.glob("out/ablate_*/log.csv"), key=os.path.getmtime):
    vals = [float(r["val_loss"]) for r in csv.DictReader(open(p)) if r["val_loss"]]
    rows.append((os.path.basename(os.path.dirname(p)).replace("ablate_", ""), vals[-1]))
with open("assets/ablations.md", "w") as f:
    f.write("| variant | final val loss |\n|---|---:|\n")
    for n, v in rows:
        f.write(f"| {n} | {v:.4f} |\n")
print(open("assets/ablations.md").read())
EOF
