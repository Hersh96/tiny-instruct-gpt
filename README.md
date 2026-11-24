# DwightGPT: a tiny instruct model built from scratch

A small GPT built from scratch in PyTorch, with every piece of the modern LLM pipeline written by hand:
a byte-level BPE tokenizer, a Llama-style transformer, a pretraining loop, supervised instruction
fine-tuning, a KV-cache for fast generation, and a web demo.

It is pretrained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and then
fine-tuned on [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)
so it follows requests like *"Write a story about a brave turtle. Use the words river and friend."*

The twist: the stories star **Dwight**, a very serious beet farmer in glasses who walks into the story and
says something deadpan and over-confident, in the spirit of Dwight Schrute from *The Office*.
You can also ask him questions directly (*"Dwight, what should I do if I see a bear?"*).
Everything trains on one consumer GPU or a free Colab T4.

<a href="https://colab.research.google.com/github/Hersh96/tiny-instruct-gpt/blob/main/colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

## What's inside

| Component | File | Notes |
|---|---|---|
| Byte-level BPE tokenizer | [`tinygpt/tokenizer.py`](tinygpt/tokenizer.py) | GPT-2 style pre-split, incremental pair counts, special chat tokens, no `<unk>` |
| Transformer | [`tinygpt/model.py`](tinygpt/model.py) | RMSNorm, RoPE, grouped-query attention, SwiGLU, fused SDPA (Flash), weight tying |
| KV-cache | [`tinygpt/model.py`](tinygpt/model.py) | pre-allocated per-layer buffers, prefill + single-token decode |
| Sampling | [`tinygpt/model.py`](tinygpt/model.py) | temperature, top-k, top-p, streaming |
| Training loop | [`train.py`](train.py) | bf16/fp16 autocast, `torch.compile`, grad accumulation, cosine LR + warmup, clipping, CSV logs |
| Instruction tuning | [`train.py --stage sft`](train.py), [`tinygpt/data.py`](tinygpt/data.py) | chat template, loss masked to the assistant's reply only |
| Dwight theme | [`tinygpt/dwight.py`](tinygpt/dwight.py) | weaves Dwight scenes into SFT stories, plus short "ask Dwight" chats |
| Ablations | [`scripts/ablate.sh`](scripts/ablate.sh) | GPT-2 block → +RMSNorm → +RoPE → +SwiGLU → +GQA |
| Benchmarks | [`bench.py`](bench.py) | training tokens/s (fp32 vs bf16 vs compile), generation with/without KV-cache |
| Web demo | [`app.py`](app.py) | Gradio chat + base-model completion, ready for Hugging Face Spaces |
| Tests | [`tests/`](tests) | tokenizer round-trip, causality, KV-cache ≡ full forward, RoPE relativity, SFT mask |

Every modern component can be switched back to its GPT-2 counterpart (`--norm layernorm --mlp gelu --pos learned --n_kv_head <n_head>`),
which is how the ablation study runs.

## Architecture

```
tokens ─► embedding ─► [ Block × n_layer ] ─► RMSNorm ─► lm_head (tied to embedding)

Block:  x ─► RMSNorm ─► Attention ─► + ─► RMSNorm ─► SwiGLU MLP ─► +
        └───────────────────────────┘ └──────────────────────────────┘
Attention: q (n_head heads), k/v (n_kv_head heads, shared across groups)
           RoPE on q and k ─► scaled_dot_product_attention(causal) ─► o_proj
```

| preset | layers | d_model | heads (q / kv) | context | params | where it trains |
|---|---:|---:|---:|---:|---:|---|
| `tiny` | 2 | 128 | 4 / 2 | 256 | 1.4M | CPU, for tests |
| `small` | 6 | 384 | 6 / 2 | 512 | 12.6M | free Colab T4, about an hour |
| `base` | 8 | 640 | 8 / 4 | 512 | 41.6M | one 16–24 GB GPU, a few hours |

## Quickstart

```bash
pip install -r requirements.txt

# 1. data + tokenizer (downloads ~600 MB of stories and ~150 MB of instructions)
python scripts/prepare.py --pretrain_mb 600 --sft_mb 150 --vocab_size 8192

# 2. pretrain
python train.py --stage pretrain --preset small

# 3. instruction-tune
python train.py --stage sft --init_from out/pretrain/ckpt.pt

# 4. talk to it
python sample.py --ckpt out/sft/ckpt.pt --chat "Write a story about a dragon who is scared of the dark. Dwight should be in it."
python sample.py --ckpt out/sft/ckpt.pt --interactive

# 5. curves, benchmarks, demo
python scripts/plot.py out/pretrain/log.csv out/sft/log.csv --out assets/loss.png
python bench.py --preset small
python app.py            # http://127.0.0.1:7860
```

Or open [`colab.ipynb`](colab.ipynb) and run every cell.

## How instruction tuning works

Each TinyStoriesInstruct record has a summary, required words, a sentence to include and story features.
[`instruct_to_chat`](tinygpt/data.py) turns a random subset of these into a natural request, and the story becomes the reply:

```
<|user|>Write a short story about this: Sara and Ben play in the park. Use the words quit, oak and gloomy. It should include some dialogue.
<|assistant|>Once upon a time, Sara and Ben were playing in the park...<|endoftext|>
```

### Adding Dwight

[`tinygpt/dwight.py`](tinygpt/dwight.py) rewrites 80% of the SFT stories (`--dwight_frac`): Dwight enters after the
opening paragraph with a line in his voice, everyone reacts, and he closes the story. Half of those requests ask
for him ("Dwight should be in the story."), and in the other half he just shows up, so he becomes the model's
default sidekick. About 1% of the data is short single-turn chats where you talk to Dwight directly.

```
<|user|>Write a short story about this: Max the bird loses a hat and finds it. Use the words hat and tree. Add Dwight to the story.
<|assistant|>Once upon a time, there was a little bird named Max. Max had a shiny hat.

Then Dwight came by. Dwight was a very serious man who grew beets and knew a lot of facts. "Question. What is the best
thing about a hat? Answer. Everything," he said. Everyone giggled. Dwight did not giggle. He nodded very seriously.

One day, Max lost the hat. ...

Dwight pushed up his glasses and said, "Another problem solved. You are welcome." The end.<|endoftext|>
```

All of Dwight's lines are original, written in his style for this project rather than quoted from the show.
They use the simple vocabulary of TinyStories, so a 13M-parameter model can learn them, and they are kid-friendly.
Edit the lists in `dwight.py` to give him new material, then rerun `prepare.py`.

### Loss masking

During SFT the loss is computed only on the reply and the closing `<|endoftext|>`, never on the
prompt or padding, so the model learns to *answer* rather than to *write requests*.
`tests/test_core.py::test_instruct_parsing_and_sft_mask` checks this.

## Results

> Fill these in from your own run: the Colab notebook writes every file below into `assets/`.

### Loss curves
![loss curves](assets/loss.png)

### Ablations (same data, steps and seed)
See `assets/ablations.png` and `assets/ablations.md` after running `bash scripts/ablate.sh`.
GQA lowers the parameter count, so read its row as "how much quality do we lose for a smaller, faster KV-cache".

### Speed
See `assets/benchmarks.md` after running `python bench.py`.

### Samples
Add a few prompts and the model's replies here.

## Running the tests

```bash
pytest -q
```

The tests run on CPU in a few seconds and include a check that cached generation produces exactly the
same tokens as recomputing the full context.

## Deploying the demo to Hugging Face Spaces

1. Create a Gradio Space.
2. Upload `app.py`, the `tinygpt/` folder, `requirements.txt`, `data/tokenizer.json`, `out/sft/ckpt.pt` and (optionally) `out/pretrain/ckpt.pt`.
3. The Space runs `app.py`, which finds the checkpoints at the default paths. Override them with the `CKPT`, `BASE_CKPT` and `TOKENIZER` environment variables.

The `small` model runs comfortably on the free CPU tier.

## References

- *The Office* (US), for Dwight Schrute, the inspiration for the theme. This is an unaffiliated fan project.
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) and [minbpe](https://github.com/karpathy/minbpe), the starting points for this style of project
- Eldan & Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (2023)
- Su et al., [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021)
- Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (2023)

## License

MIT
