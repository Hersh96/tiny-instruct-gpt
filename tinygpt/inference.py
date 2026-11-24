"""Load a checkpoint and stream text or chat replies."""

from __future__ import annotations

import os
from typing import Iterator

import torch

from .data import format_chat
from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer


class Generator:
    def __init__(self, ckpt_path: str, tokenizer_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = GPTConfig(**ckpt["config"])
        self.model = GPT(self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.stage = ckpt.get("stage", "pretrain")
        self.tok = BPETokenizer.load(tokenizer_path)
        self.stop = {self.tok.eot_id, self.tok.special["<|user|>"]}

    def stream(self, prompt_ids: list[int], max_new_tokens=300, temperature=0.8, top_k=50, top_p=0.95) -> Iterator[str]:
        """Yields the decoded text so far, handling multi-byte UTF-8 safely."""
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out: list[int] = []
        for nxt in self.model.generate(idx, max_new_tokens, temperature, top_k, top_p, stop_ids=self.stop):
            t = nxt.item()
            if t in self.stop:
                break
            out.append(t)
            yield self.tok.decode(out)

    def complete(self, text: str, **kw) -> Iterator[str]:
        ids = [self.tok.eot_id] + self.tok.encode_ordinary(text)
        for s in self.stream(ids, **kw):
            yield text + s

    def chat(self, message: str, **kw) -> Iterator[str]:
        # Single-turn: this model was tuned on one request -> one story.
        ids = self.tok.encode(format_chat(message))
        yield from self.stream(ids, **kw)


def default_paths(stage: str = "sft") -> tuple[str, str]:
    return os.path.join("out", stage, "ckpt.pt"), os.path.join("data", "tokenizer.json")
