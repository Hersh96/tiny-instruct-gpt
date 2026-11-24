import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygpt.data import SFTLoader, format_chat, instruct_to_chat, parse_instruct  # noqa: E402
from tinygpt.model import GPT, GPTConfig, apply_rope, rope_cache  # noqa: E402
from tinygpt.tokenizer import BPETokenizer  # noqa: E402

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside. "
    "One day, Lily saw a big red ball in the park. \"Can I play?\" she asked. "
) * 50


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train([TEXT], vocab_size=400)
    return t


def test_tokenizer_roundtrip(tok):
    for s in [TEXT[:300], "héllo wörld 🙂 123456", "  spaces\n\nnewlines\t", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_tokenizer_compresses(tok):
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 2


def test_special_tokens(tok, tmp_path):
    s = format_chat("hi", "hello")
    ids = tok.encode(s)
    assert tok.special["<|user|>"] in ids and tok.special["<|assistant|>"] in ids and ids[-1] == tok.eot_id
    assert tok.special["<|user|>"] not in tok.encode_ordinary(s)  # ordinary text never yields specials
    p = tmp_path / "tok.json"
    tok.save(str(p))
    tok2 = BPETokenizer.load(str(p))
    assert tok2.encode(s) == ids and tok2.vocab_size == tok.vocab_size


def small_cfg(**kw):
    return GPTConfig(**{**dict(vocab_size=128, block_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64), **kw})


@pytest.mark.parametrize("kw", [{}, dict(norm="layernorm", mlp="gelu", pos="learned", n_kv_head=4)])
def test_causality(kw):
    torch.manual_seed(0)
    m = GPT(small_cfg(**kw)).eval()
    x = torch.randint(0, 128, (1, 20))
    x2 = x.clone()
    x2[0, 15:] = torch.randint(0, 128, (5,))
    l1, _ = m(x, x)
    l2, _ = m(x2, x2)
    assert torch.allclose(l1[0, :15], l2[0, :15], atol=1e-5)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(small_cfg()).eval()
    x = torch.randint(0, 128, (2, 24))
    full, _ = m(x, x)  # logits for every position
    cache = m.new_cache(2)
    logits_prefill, _ = m(x[:, :10], cache=cache)
    assert torch.allclose(logits_prefill[:, -1], full[:, 9], atol=1e-5)
    for t in range(10, 24):
        step, _ = m(x[:, t : t + 1], cache=cache)
        assert torch.allclose(step[:, -1], full[:, t], atol=1e-4)


def test_generate_cache_equals_no_cache():
    torch.manual_seed(0)
    m = GPT(small_cfg()).eval()
    idx = torch.randint(0, 128, (1, 5))
    a = torch.cat(list(m.generate(idx, 20, temperature=0, use_cache=True)), 1)
    b = torch.cat(list(m.generate(idx, 20, temperature=0, use_cache=False)), 1)
    assert torch.equal(a, b)


def test_rope_is_relative():
    # <R(m)q, R(n)k> depends only on m - n
    cos, sin = rope_cache(32, 16, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    def dot(m, n):
        return (apply_rope(q, cos[m : m + 1], sin[m : m + 1]) * apply_rope(k, cos[n : n + 1], sin[n : n + 1])).sum()
    assert torch.allclose(dot(5, 2), dot(20, 17), atol=1e-5)


def test_swiglu_param_parity():
    a = GPT(small_cfg(mlp="swiglu", n_embd=192, n_kv_head=4)).num_params()
    b = GPT(small_cfg(mlp="gelu", n_embd=192, n_kv_head=4)).num_params()
    assert abs(a - b) / b < 0.03


def test_overfit_single_batch():
    torch.manual_seed(0)
    m = GPT(small_cfg())
    opt = m.configure_optimizer(3e-3, 0.0)
    x = torch.randint(0, 128, (4, 32))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.1


def test_instruct_parsing_and_sft_mask(tok, tmp_path):
    doc = ("Features: Dialogue\nWords: quit, oak, gloomy\nSummary: Sara and Ben play in the park.\n"
           "Story: \n\nOnce upon a time, Sara and Ben were playing in the park under a big oak tree.")
    rec = parse_instruct(doc)
    assert rec["Words"] == "quit, oak, gloomy" and rec["Story"].startswith("Once upon a time")
    import random
    prompt, reply = instruct_to_chat(rec, random.Random(0))
    p_ids = tok.encode(format_chat(prompt))
    r_ids = tok.encode_ordinary(reply) + [tok.eot_id]
    seq = np.asarray(p_ids + r_ids, np.uint16)
    np.savez(tmp_path / "s.npz", tokens=seq, offsets=np.array([0, len(seq)]), prompt_lens=np.array([len(p_ids)]))
    dl = SFTLoader(str(tmp_path / "s.npz"), 1, 256, "cpu", tok.special["<|pad|>"])
    x, y, mask = dl.batch()
    targets = y[0][mask[0]].tolist()
    assert targets == r_ids  # loss only on the reply + <|endoftext|>
