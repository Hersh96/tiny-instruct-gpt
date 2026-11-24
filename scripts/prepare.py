"""Download TinyStories, train the BPE tokenizer, and tokenize everything.

Outputs (in --data_dir):
  tokenizer.json            learned BPE merges
  pretrain_{train,val}.bin  flat uint16 token streams, documents joined by <|endoftext|>
  sft_{train,val}.npz       chat examples + prompt lengths for loss masking

Example:
  python scripts/prepare.py --pretrain_mb 600 --sft_mb 200 --vocab_size 8192
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygpt.data import download, format_chat, instruct_to_chat, parse_instruct, read_documents  # noqa: E402
from tinygpt.dwight import QA as DWIGHT_QA  # noqa: E402
from tinygpt.tokenizer import BPETokenizer  # noqa: E402

_TOK: BPETokenizer | None = None


def _init(path: str) -> None:
    global _TOK
    _TOK = BPETokenizer.load(path)


def _encode_doc(doc: str) -> list[int]:
    return _TOK.encode_ordinary(doc) + [_TOK.eot_id]


def _encode_chat(pair: tuple[str, str]) -> tuple[list[int], int]:
    prompt, reply = pair
    p_ids = _TOK.encode(format_chat(prompt))
    r_ids = _TOK.encode_ordinary(reply.strip()) + [_TOK.eot_id]
    return p_ids + r_ids, len(p_ids)


def encode_pretrain(docs, tok_path, out_path, workers):
    t0 = time.time()
    arr_parts, n = [], 0
    with Pool(workers, initializer=_init, initargs=(tok_path,)) as pool:
        for ids in pool.imap(_encode_doc, docs, chunksize=256):
            arr_parts.append(np.asarray(ids, dtype=np.uint16))
            n += len(ids)
    arr = np.concatenate(arr_parts)
    arr.tofile(out_path)
    print(f"  {out_path}: {n / 1e6:.1f}M tokens from {len(docs):,} docs ({time.time() - t0:.0f}s)")


def encode_sft(pairs, tok_path, out_path, workers, block_size):
    toks, offsets, plens = [], [0], []
    with Pool(workers, initializer=_init, initargs=(tok_path,)) as pool:
        for ids, plen in pool.imap(_encode_chat, pairs, chunksize=256):
            if len(ids) > block_size + 1 or plen >= len(ids) - 1:
                continue  # skip examples that would be truncated mid-reply
            toks.append(np.asarray(ids, dtype=np.uint16))
            offsets.append(offsets[-1] + len(ids))
            plens.append(plen)
    np.savez(out_path, tokens=np.concatenate(toks), offsets=np.asarray(offsets, np.int64), prompt_lens=np.asarray(plens, np.int32))
    print(f"  {out_path}: {len(plens):,} chat examples (kept of {len(pairs):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--tokenizer_mb", type=float, default=30, help="MB of text used to learn merges")
    ap.add_argument("--pretrain_mb", type=int, default=600, help="MB of pretraining text to download (0 = all ~2.2GB)")
    ap.add_argument("--sft_mb", type=int, default=150, help="MB of instruct text to download (0 = all ~2.7GB)")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--dwight_frac", type=float, default=0.8, help="share of SFT stories with Dwight in them (0 = plain)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--local_text", default=None, help="skip downloads; use this text file (docs split by <|endoftext|>)")
    ap.add_argument("--local_instruct", default=None, help="with --local_text: an instruct-format text file for SFT")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    rng = random.Random(1337)

    # ---- raw text
    if args.local_text:
        docs = read_documents(args.local_text)
        rng.shuffle(docs)
        k = max(1, len(docs) // 50)
        train_docs, val_docs = docs[k:], docs[:k]
        sft_train_raw, sft_val_raw = [], []
        if args.local_instruct:
            raw = read_documents(args.local_instruct)
            k = max(1, len(raw) // 50)
            sft_train_raw, sft_val_raw = raw[k:], raw[:k]
    else:
        mb = lambda x: x * 1_000_000 if x else None  # noqa: E731
        train_docs = read_documents(download("pretrain-train", args.data_dir, mb(args.pretrain_mb)))
        val_docs = read_documents(download("pretrain-valid", args.data_dir))
        sft_train_raw = read_documents(download("instruct-train", args.data_dir, mb(args.sft_mb)))
        sft_val_raw = read_documents(download("instruct-valid", args.data_dir))
    print(f"pretrain docs: {len(train_docs):,} train / {len(val_docs):,} val")

    # ---- tokenizer
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        sample, size = [], 0
        for d in train_docs:
            sample.append(d)
            size += len(d)
            if size > args.tokenizer_mb * 1e6:
                break
        print(f"training BPE (vocab {args.vocab_size}) on {size / 1e6:.1f} MB ...")
        t0 = time.time()
        tok = BPETokenizer()
        tok.train(sample, args.vocab_size, verbose=True)
        tok.save(tok_path)
        print(f"  done in {time.time() - t0:.0f}s")
    tok = BPETokenizer.load(tok_path)
    demo = "Once upon a time, there was a little girl named Lily."
    print(f"tokenizer: vocab={tok.vocab_size}, '{demo}' -> {len(tok.encode(demo))} tokens")

    # ---- pretraining stream
    print("encoding pretraining data ...")
    encode_pretrain(train_docs, tok_path, os.path.join(args.data_dir, "pretrain_train.bin"), args.workers)
    encode_pretrain(val_docs, tok_path, os.path.join(args.data_dir, "pretrain_val.bin"), args.workers)

    # ---- SFT
    if sft_train_raw:
        print("building SFT chat data ...")
        for raw, split in ((sft_train_raw, "train"), (sft_val_raw, "val")):
            recs = [r for r in map(parse_instruct, raw) if r]
            pairs = [instruct_to_chat(r, rng, args.dwight_frac) for r in recs]
            if args.dwight_frac > 0:
                # Add the short "ask Dwight" chats at ~1% of the data.
                reps = max(1, len(pairs) // (100 * len(DWIGHT_QA)))
                pairs += [qa for qa in DWIGHT_QA for _ in range(reps)]
                rng.shuffle(pairs)
            if split == "train":
                ex = next((p for p in pairs if "Dwight" in p[0] and len(p[1]) > 200), pairs[0])
                print(f"  example:\n  USER: {ex[0]}\n  ASSISTANT: {ex[1][:600]}")
            encode_sft(pairs, tok_path, os.path.join(args.data_dir, f"sft_{split}.npz"), args.workers, args.block_size)


if __name__ == "__main__":
    main()
