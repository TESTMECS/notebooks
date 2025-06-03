"""SpookyBench Time-Aware Decoder
--------------------------------
Example project showing how to train a time-aware decoder using the
GravityField and causal projection utilities from :mod:`TODO.combined`.

The SpookyBench dataset is loaded from Hugging Face. Each text sample
is embedded with a transformer model and refined with a learned
``GravityField``. The ``RealNVPDecoder`` consumes these causally
ordered embeddings using an invertible flow network.
"""

from __future__ import annotations

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from TODO.combined import GravityField, causal_refine_embeddings_with_phi

from .realnvp import RealNVPDecoder


def load_spookybench(split: str = "train[:10]"):
    """Load a small portion of the SpookyBench text split."""
    return load_dataset("timeblindness/spooky-bench", name="text", split=split)


def process_sample(tokenizer, model, phi_model, text: str):
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens, output_attentions=True)
    emb = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
    attn = outputs.attentions[-1][0].mean(0)  # average heads
    _, refined = causal_refine_embeddings_with_phi(emb, attn, phi_model)
    return refined


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased")
    phi_model = GravityField(input_dim=4)
    decoder = RealNVPDecoder(input_dim=769)

    dataset = load_spookybench()
    for item in dataset:
        refined = process_sample(tokenizer, bert, phi_model, item["label_text_content"])
        pred = decoder(refined)
        print(pred.shape)


if __name__ == "__main__":
    main()
