# Recent Changes and Accomplishments

This repository has been expanded with a focus on causal embeddings and invertible neural networks.

## RealNVP Flow and Decoder
- Implemented an invertible RealNVP module that alternates coupling layers for stable round‑trip transformations.
- Wrapped the flow in a `RealNVPDecoder` so it can be used in place of a standard feed‑forward decoder.
- Added unit tests (`tests/test_realnvp.py`) to verify round‑trip correctness and decoder output shape.

## Causal Projection Utilities
- Introduced functions in `TODO/combined.py` for projecting embeddings onto a causal manifold using a learnable `GravityField`.
- Added tests (`tests/test_phi_utils.py`) to ensure the field outputs positive values and that projection enforces a negative Minkowski interval.

## SpookyBench Demonstration
- Created `spookybench_project` showcasing a small pipeline that trains a time‑aware decoder on a subset of the SpookyBench dataset.
- Documentation (`spookybench_project/README.md`) explains how the data are loaded, refined using `GravityField`, and passed to the decoder.

These additions provide a starting point for experimenting with time-aware attention models and causal embedding refinements.
