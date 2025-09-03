
# XAI for Clustering — Spectral Graphs & Shortest-Path Attributions

**Goal.** Explain spectral clustering assignments with graph-theoretic attributions using shortest paths
over k-NN graphs built from embeddings. Datasets: **MNIST** embeddings, **COIL-20**, **Spherical Gaussians (synthetic)**.

## Quickstart
```bash
make init
python -m src.pipeline --dataset mnist --k 15
```
