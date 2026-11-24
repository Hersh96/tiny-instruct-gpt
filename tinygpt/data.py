"""Dataset download, parsing, chat formatting and batch loading."""

from __future__ import annotations

import os
import random
import re
import sys
import urllib.request

import numpy as np
import torch

HF = "https://huggingface.co/datasets/roneneldan"
URLS = {
    "pretrain-train": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "pretrain-valid": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "instruct-train": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt",
    "instruct-valid": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt",
}
EOT = "<|endoftext|>"
USER, ASSISTANT = "<|user|>", "<|assistant|>"


def download(name: str, data_dir: str, max_bytes: int | None = None) -> str:
    """Download (optionally only the first `max_bytes`) of a dataset file."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, os.path.basename(URLS[name]))
    if max_bytes:
        path += f".{max_bytes // 1_000_000}MB"
    if os.path.exists(path):
        return path
    print(f"downloading {URLS[name]} -> {path}")
    req = urllib.request.Request(URLS[name])
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = path + ".part"
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB")
                sys.stdout.flush()
    print()
    os.replace(tmp, path)
    return path


def read_documents(path: str) -> list[str]:
    """Split a TinyStories text file into documents, dropping a truncated tail."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    docs = [d.strip() for d in text.split(EOT)]
    if not text.rstrip().endswith(EOT):
        docs = docs[:-1]  # last document was cut off by a partial download
    return [d for d in docs if d]


# ------------------------------------------------------------- chat format
def format_chat(prompt: str, response: str | None = None) -> str:
    s = f"{USER}{prompt.strip()}\n{ASSISTANT}"
    if response is not None:
        s += response.strip() + EOT
    return s


_FIELD = re.compile(r"^(Features|Words|Summary|Random sentence|Story):\s*(.*)$")


def parse_instruct(doc: str) -> dict | None:
    fields, cur = {}, None
    for line in doc.splitlines():
        m = _FIELD.match(line.strip())
        if m and cur != "Story":
            cur = m.group(1)
            fields[cur] = m.group(2)
        elif cur:
            fields[cur] = (fields[cur] + "\n" + line).strip()
    story = fields.get("Story", "").strip()
    return {**fields, "Story": story} if len(story) > 50 else None


FEATURE_TEXT = {
    "Dialogue": "include some dialogue",
    "BadEnding": "have a sad ending",
    "MoralValue": "teach a lesson",
    "Foreshadowing": "use foreshadowing",
    "Conflict": "have a conflict",
    "Twist": "have a plot twist",
}


def instruct_to_chat(rec: dict, rng: random.Random, dwight_frac: float = 0.0) -> tuple[str, str]:
    """Turn a TinyStoriesInstruct record into a natural-language request.

    With probability `dwight_frac` the reply story gets Dwight woven in (see
    tinygpt/dwight.py). Half of those requests ask for him by name; in the
    rest he just shows up, so he becomes the model's default sidekick.
    """
    parts = []
    if rec.get("Summary") and rng.random() < 0.6:
        parts.append(f"Write a short story about this: {rec['Summary']}")
    else:
        parts.append(rng.choice(["Write a short story.", "Tell me a story.", "Can you write a story for a child?"]))
    if rec.get("Words") and rng.random() < 0.7:
        words = [w.strip() for w in rec["Words"].split(",") if w.strip()]
        if words:
            joined = ", ".join(words[:-1]) + (" and " if len(words) > 1 else "") + words[-1]
            parts.append(f"Use the words {joined}.")
    if rec.get("Random sentence") and rng.random() < 0.5:
        parts.append(f'Include the sentence: "{rec["Random sentence"]}"')
    feats = [FEATURE_TEXT[f.strip()] for f in rec.get("Features", "").split(",") if f.strip() in FEATURE_TEXT]
    if feats and rng.random() < 0.7:
        parts.append("It should " + " and ".join(feats) + ".")
    story = rec["Story"]
    if rng.random() < dwight_frac:
        from .dwight import add_dwight, ask_for_dwight

        if rng.random() < 0.5:  # half the time he is asked for, half he just shows up
            parts.insert(rng.randint(1, len(parts)), ask_for_dwight(rng))
        story = add_dwight(story, rec.get("Words"), rng)
    return " ".join(parts), story


# ---------------------------------------------------------------- loaders
class PretrainLoader:
    """Random contiguous windows from a flat uint16 token file."""

    def __init__(self, path: str, batch_size: int, block_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T, self.device = batch_size, block_size, device

    def batch(self):
        ix = torch.randint(len(self.data) - self.T - 1, (self.B,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.T].astype(np.int64)) for i in ix])
        return _to(x, self.device), _to(y, self.device), None


class SFTLoader:
    """Variable-length chat examples padded to the batch max.

    Loss is computed only on the assistant's reply (and the closing
    <|endoftext|>), never on the user prompt or padding.
    """

    def __init__(self, path: str, batch_size: int, block_size: int, device: str, pad_id: int):
        blob = np.load(path)
        self.tokens, self.offsets, self.prompt_lens = blob["tokens"], blob["offsets"], blob["prompt_lens"]
        self.n = len(self.prompt_lens)
        self.B, self.T, self.device, self.pad = batch_size, block_size, device, pad_id

    def batch(self):
        ix = np.random.randint(0, self.n, self.B)
        seqs = []
        for i in ix:
            seq = self.tokens[self.offsets[i] : self.offsets[i + 1]][: self.T + 1].astype(np.int64)
            seqs.append((seq, int(self.prompt_lens[i])))
        L = max(len(s) for s, _ in seqs) - 1
        L = min(self.T, 64 * ((L + 63) // 64))  # few distinct shapes -> few torch.compile recompiles
        x = torch.full((self.B, L), self.pad, dtype=torch.long)
        y = torch.full((self.B, L), -1, dtype=torch.long)
        mask = torch.zeros((self.B, L), dtype=torch.bool)
        for b, (seq, p) in enumerate(seqs):
            n = len(seq) - 1
            x[b, :n] = torch.from_numpy(seq[:-1])
            y[b, :n] = torch.from_numpy(seq[1:])
            mask[b, max(p - 1, 0) : n] = True  # predict tokens from the reply onward
        y[~mask] = -1
        return _to(x, self.device), _to(y, self.device), _to(mask, self.device)


def _to(t: torch.Tensor, device: str) -> torch.Tensor:
    if device.startswith("cuda"):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)
