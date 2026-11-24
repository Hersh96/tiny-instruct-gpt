"""Command-line sampling.

  python sample.py --ckpt out/pretrain/ckpt.pt --prompt "Once upon a time"
  python sample.py --ckpt out/sft/ckpt.pt --chat "Write a story about a brave cat."
  python sample.py --ckpt out/sft/ckpt.pt --interactive
"""

import argparse
import sys

from tinygpt.inference import Generator


def run(gen_iter):
    prev = ""
    for text in gen_iter:
        sys.stdout.write(text[len(prev):])
        sys.stdout.flush()
        prev = text
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/sft/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default=None, help="plain text completion")
    ap.add_argument("--chat", default=None, help="one instruction for the SFT model")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--num_samples", type=int, default=1)
    args = ap.parse_args()

    g = Generator(args.ckpt, args.tokenizer)
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    if args.interactive:
        while True:
            try:
                msg = input("\nyou> ")
            except (EOFError, KeyboardInterrupt):
                break
            sys.stdout.write("bot> ")
            run(g.chat(msg, **kw))
        return
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        if args.chat:
            run(g.chat(args.chat, **kw))
        else:
            run(g.complete(args.prompt or "Once upon a time", **kw))


if __name__ == "__main__":
    main()
