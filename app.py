"""Gradio web demo. Works locally and as a Hugging Face Space.

  pip install gradio
  python app.py                      # uses out/sft/ckpt.pt and data/tokenizer.json
  python app.py --ckpt path/to/ckpt.pt --share
"""

import argparse
import os

import gradio as gr

from tinygpt.inference import Generator

EXAMPLES = [
    "Write a short story about a little dog who is afraid of the rain. Dwight should be in the story.",
    "Tell me a story. Use the words moon, jump and brave.",
    "Write a short story about this: Tom loses his red ball and his sister helps him find it. Add Dwight to the story.",
    "Dwight, what should I do if I see a bear?",
]


def build(gen: Generator, base: Generator | None):
    def respond(message, history, temperature, top_p, max_tokens):
        kw = dict(temperature=temperature, top_p=top_p, top_k=0, max_new_tokens=int(max_tokens))
        yield from gen.chat(message, **kw)

    def complete(prompt, temperature, top_p, max_tokens):
        kw = dict(temperature=temperature, top_p=top_p, top_k=0, max_new_tokens=int(max_tokens))
        yield from (base or gen).complete(prompt, **kw)

    n_params = sum(p.numel() for p in gen.model.parameters()) / 1e6
    with gr.Blocks(title="DwightGPT") as demo:
        gr.Markdown(
            f"# DwightGPT 🥕\nA {n_params:.1f}M-parameter GPT written and trained from scratch "
            "(own BPE tokenizer, RoPE, RMSNorm, SwiGLU, GQA, KV-cache), pretrained on TinyStories "
            "and instruction-tuned to write children's stories on request, starring Dwight, a very serious beet farmer."
        )
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top-p")
            max_tokens = gr.Slider(32, 480, value=300, step=16, label="max new tokens")
        with gr.Tab("Chat (instruct model)"):
            gr.ChatInterface(respond, examples=[[e, 0.8, 0.95, 300] for e in EXAMPLES], additional_inputs=[temperature, top_p, max_tokens],
                             cache_examples=False)
        with gr.Tab("Complete (base model)"):
            inp = gr.Textbox(value="Once upon a time, there was a", label="prompt")
            out = gr.Textbox(label="completion", lines=12)
            gr.Button("Generate").click(complete, [inp, temperature, top_p, max_tokens], out)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("CKPT", "out/sft/ckpt.pt"))
    ap.add_argument("--base_ckpt", default=os.environ.get("BASE_CKPT", "out/pretrain/ckpt.pt"))
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", "data/tokenizer.json"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    gen = Generator(args.ckpt, args.tokenizer)
    base = Generator(args.base_ckpt, args.tokenizer) if os.path.exists(args.base_ckpt) else None
    build(gen, base).queue().launch(share=args.share)
