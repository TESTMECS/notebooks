# SpookyBench Project

This folder demonstrates a minimal pipeline that trains a time-aware
decoder using the ideas in `TODO/time_aware_attention.py` and
`TODO/combined.py`.

Steps performed by `time_decoder.py`:

1. Load a small portion of the SpookyBench dataset from Hugging Face.
2. Obtain BERT embeddings and attention weights for each text sample.
3. Refine the embeddings with a learned `GravityField` using
   `causal_refine_embeddings_with_phi`.
4. Feed the refined embeddings into a simple `TimeAwareDecoder` for
   prediction.

Due to the size of the full dataset, only the first ten training
examples are loaded by default. Modify the `load_spookybench` function
if you wish to use different splits.
