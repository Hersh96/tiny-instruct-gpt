"""Plot loss curves from one or more training logs.

  python scripts/plot.py out/pretrain/log.csv out/sft/log.csv --out assets/loss.png
  python scripts/plot.py out/ablate_*/log.csv --val_only --out assets/ablations.png
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    train, val = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            it = int(r["iter"])
            if r["val_loss"]:
                val.append((it, float(r["val_loss"])))
            elif r["train_loss"]:
                train.append((it, float(r["train_loss"])))
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="assets/loss.png")
    ap.add_argument("--val_only", action="store_true")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    n = 1 if args.val_only else len(args.logs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for i, path in enumerate(args.logs):
        name = os.path.basename(os.path.dirname(path)) or path
        train, val = load(path)
        ax = axes[0][0 if args.val_only else i]
        if not args.val_only and train:
            ax.plot(*zip(*train), alpha=0.35, linewidth=1, label="train (per step)")
        if val:
            ax.plot(*zip(*val), marker="o", markersize=3, label=f"{name} val" if args.val_only else "val")
        ax.set_xlabel("iteration")
        ax.set_ylabel("cross-entropy loss")
        ax.set_title(args.title or ("validation loss" if args.val_only else name))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
