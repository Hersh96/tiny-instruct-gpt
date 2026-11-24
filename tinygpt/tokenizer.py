"""Byte-level BPE tokenizer, written from scratch.

How it works
------------
1. Text is split into "chunks" with a GPT-2 style regex, so merges never cross
   word / punctuation / whitespace boundaries.
2. Every chunk starts as a sequence of raw UTF-8 bytes (ids 0..255), so any
   string can be encoded and there is never an <unk> token.
3. Training repeatedly finds the most frequent adjacent pair of ids across all
   chunks (weighted by how often each chunk occurs) and merges it into a new id.
4. Special tokens (<|endoftext|>, <|user|>, <|assistant|>) are appended after
   the learned merges and are never produced by the BPE itself.

Training works on unique chunks with counts rather than the raw text, which is
what makes a pure-Python implementation fast enough for a few tens of MB.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

# GPT-2 style pre-tokenization, restricted to what Python's `re` supports.
SPLIT_PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]


def _pair_counts(words: dict[tuple[int, ...], int]) -> Counter:
    counts: Counter = Counter()
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] += freq
    return counts


def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out, i, n = [], 0, len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}
        self._cache: dict[str, list[int]] = {}
        self._special_re: re.Pattern | None = None

    # ------------------------------------------------------------------ train
    def train(self, texts: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges > 0, "vocab_size too small"

        chunk_counts: Counter = Counter()
        for text in texts:
            chunk_counts.update(SPLIT_PATTERN.findall(text))
        words = {tuple(c.encode("utf-8")): f for c, f in chunk_counts.items()}

        # Incremental bookkeeping: pair -> count, and pair -> set of words containing it.
        pair_counts = _pair_counts(words)
        where: dict[tuple[int, int], set] = {}
        for w in words:
            for p in zip(w, w[1:]):
                where.setdefault(p, set()).add(w)

        for step in range(n_merges):
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[pair] < 2:
                break
            new_id = 256 + step
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            for w in list(where.get(pair, ())):
                if w not in words:
                    continue
                freq = words.pop(w)
                for p in zip(w, w[1:]):
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    s = where.get(p)
                    if s is not None:
                        s.discard(w)
                nw = _merge(w, pair, new_id)
                words[nw] = words.get(nw, 0) + freq
                for p in zip(nw, nw[1:]):
                    pair_counts[p] += freq
                    where.setdefault(p, set()).add(nw)
            where.pop(pair, None)
            pair_counts.pop(pair, None)

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}: {self.vocab[new_id]!r}")

        self._add_specials()

    def _add_specials(self) -> None:
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, idx in self.special.items():
            self.vocab[idx] = tok.encode("utf-8")
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in self.special) + ")")
        self._cache.clear()

    # ----------------------------------------------------------------- encode
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    @property
    def eot_id(self) -> int:
        return self.special["<|endoftext|>"]

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Merge the pair that was learned earliest (lowest new id).
            best = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = list(_merge(tuple(ids), best, self.merges[best]))
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or not self.special:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special:
                out.append(self.special[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        data = {"merges": [[a, b, i] for (a, b), i in self.merges.items()], "special": SPECIAL_TOKENS}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        for a, b, i in sorted(data["merges"], key=lambda m: m[2]):
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        tok._add_specials()
        return tok
