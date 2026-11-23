## Quality at the training length

Mean ± std across seeds (sample std; no ± means a single run). Val loss is measured on 3,276,800 held-out FineWeb-Edu tokens at the training context.

| Method | Seeds | Cache vs GQA | Params | Non-emb params | Val loss | Δ vs GQA | Val PPL | HellaSwag | WinoG | OBQA | ARC-E | ARC-C | **Avg (5, paper)** | PIQA | LAMBADA acc | LAMBADA ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GQA (baseline) | 3 | 100.0% | 48.31M | 22.55M | 3.4810 ± 0.0025 | - | 32.5 ± 0.1 | 27.5 ± 0.2 | 50.5 ± 0.9 | 27.7 ± 0.5 | 41.5 ± 0.8 | 23.5 ± 0.8 | **34.12 ± 0.29** | 58.5 ± 1.2 | 16.5 ± 0.4 | 588 ± 52 |
| MQA (GQA, G=1) | 3 | 50.0% | 47.78M | 22.03M | 3.5068 ± 0.0092 | +0.0259 | 33.3 ± 0.3 | 27.6 ± 0.2 | 50.7 ± 1.2 | 27.5 ± 0.6 | 42.1 ± 0.7 | 23.5 ± 0.3 | **34.27 ± 0.51** | 58.3 ± 0.8 | 15.4 ± 1.0 | 692 ± 84 |
| MLA (dc=128, DRoPE16) | 3 | 56.2% | 49.42M | 23.67M | 3.4961 ± 0.0102 | +0.0151 | 33.0 ± 0.3 | 27.5 ± 0.3 | 50.9 ± 1.5 | 27.1 ± 0.2 | 40.8 ± 0.5 | 23.1 ± 0.3 | **33.87 ± 0.23** | 58.3 ± 0.5 | 16.3 ± 0.9 | 668 ± 76 |
| GVA + DRoPE16 | 3 | 56.2% | 48.64M | 22.88M | 3.5090 ± 0.0115 | +0.0281 | 33.4 ± 0.4 | 27.6 ± 0.0 | 52.1 ± 0.3 | 26.9 ± 0.4 | 40.8 ± 0.6 | 22.8 ± 0.3 | **34.02 ± 0.20** | 58.3 ± 0.4 | 15.3 ± 0.1 | 744 ± 36 |
| GVA (RoPE, no absorb) | 3 | 50.0% | 48.05M | 22.29M | 3.5288 ± 0.0045 | +0.0478 | 34.1 ± 0.2 | 27.3 ± 0.0 | 51.9 ± 1.2 | 27.7 ± 1.6 | 41.0 ± 1.0 | 23.4 ± 0.9 | **34.26 ± 0.47** | 58.6 ± 0.4 | 15.2 ± 0.4 | 733 ± 37 |
| GKA (ours, RoPE) | 3 | 50.0% | 48.05M | 22.29M | 3.5420 ± 0.0054 | +0.0610 | 34.5 ± 0.2 | 27.4 ± 0.2 | 50.8 ± 0.6 | 27.9 ± 0.4 | 40.5 ± 1.2 | 23.0 ± 0.4 | **33.90 ± 0.17** | 58.2 ± 0.5 | 14.5 ± 0.6 | 810 ± 81 |
| GKA + cross-group V | 3 | 50.0% | 48.31M | 22.55M | 3.5315 ± 0.0056 | +0.0506 | 34.2 ± 0.2 | 27.2 ± 0.2 | 51.5 ± 0.4 | 27.5 ± 1.5 | 41.3 ± 0.4 | 23.6 ± 0.5 | **34.24 ± 0.38** | 58.5 ± 0.5 | 15.7 ± 0.5 | 763 ± 47 |
| GKA + rotated V | 3 | 50.0% | 48.05M | 22.29M | 3.5706 ± 0.0051 | +0.0896 | 35.5 ± 0.2 | 27.4 ± 0.2 | 51.5 ± 1.4 | 26.5 ± 0.5 | 40.2 ± 0.8 | 23.5 ± 0.6 | **33.86 ± 0.41** | 58.6 ± 0.3 | 13.4 ± 0.3 | 1102 ± 115 |
| GKA + cross-group + rotated V | 3 | 50.0% | 48.31M | 22.55M | 3.5403 ± 0.0063 | +0.0594 | 34.5 ± 0.2 | 27.4 ± 0.1 | 52.0 ± 0.6 | 27.3 ± 0.2 | 41.0 ± 0.8 | 23.6 ± 0.2 | **34.25 ± 0.25** | 58.0 ± 0.5 | 14.6 ± 0.3 | 902 ± 86 |
| Shared KV (K=V) | 1 | 50.0% | 47.78M | 22.03M | 3.5585 | +0.0775 | 35.1 | 27.3 | 52.8 | 26.0 | 41.4 | 22.4 | **33.97** | 57.9 | 13.9 | 1036 |

Accuracies in %. HellaSwag/ARC/OBQA/PIQA use length-normalised accuracy (acc_norm); WinoGrande uses partial scoring. Avg (5, paper) is the unweighted mean of the GVA paper's five tasks, averaged per seed. Chance: HellaSwag 25%, WinoG 50%, OBQA 25%, ARC-E 25%, ARC-C 25%, PIQA 50%.

## Inference (NVIDIA A100-SXM4-40GB, bf16, PyTorch eager, 64 decoded tokens)

| Method | KV bytes/token (all layers) | cache MB B=1 T=1024 | cache MB B=1 T=4096 | cache MB B=16 T=4096 | cache MB B=32 T=4096 |
|---|---|---|---|---|---|
| GQA (baseline) | 4096 | 4.0 | 16.0 | 256.0 | 512.0 |
| MQA (GQA, G=1) | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |
| MLA (dc=128, DRoPE16) | 2304 | 2.2 | 9.0 | 144.0 | 288.0 |
| GVA + DRoPE16 | 2304 | 2.2 | 9.0 | 144.0 | 288.0 |
| GVA (RoPE, no absorb) | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |
| GKA (ours, RoPE) | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |
| GKA + cross-group V | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |
| GKA + rotated V | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |
| GKA + cross-group + rotated V | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |
| Shared KV (K=V) | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |

| Method | decode tok/s B=1 T=1024 | decode tok/s B=1 T=4096 | decode tok/s B=16 T=4096 | decode tok/s B=32 T=4096 | prefill ms B=1 T=1024 | prefill ms B=1 T=4096 | prefill ms B=16 T=4096 | prefill ms B=32 T=4096 |
|---|---|---|---|---|---|---|---|---|
| GQA (baseline) | 97 | 97 | 1496 | 3003 | 23 | 92 | 561 | 1113 |
| MQA (GQA, G=1) | 97 | 96 | 1493 | 3045 | 23 | 92 | 569 | 1106 |
| MLA (dc=128, DRoPE16) | 80 | 80 | 1240 | 2497 | 27 | 109 | 663 | 1312 |
| GVA + DRoPE16 | 84 | 84 | 1272 | 2586 | 26 | 104 | 649 | 1280 |
| GVA (RoPE, no absorb) | 94 | 95 | 1168 | 1410 | 23 | 93 | 659 | 1555 |
| GKA (ours, RoPE) | 104 | 105 | 1596 | 2851 | 21 | 84 | 525 | 1064 |
| GKA + cross-group V | 98 | 99 | 1492 | 2660 | 22 | 88 | 548 | 1109 |
| GKA + rotated V | 86 | 85 | 1307 | 2359 | 25 | 113 | 678 | 1361 |
| GKA + cross-group + rotated V | 79 | 77 | 1188 | 2121 | 28 | 119 | 707 | 1421 |
| Shared KV (K=V) | 104 | 104 | 1590 | 2830 | 21 | 85 | 524 | 1063 |

| Method | decode peak MB B=1 T=1024 | decode peak MB B=1 T=4096 | decode peak MB B=16 T=4096 | decode peak MB B=32 T=4096 |
|---|---|---|---|---|
| GQA (baseline) | 13 | 16 | 264 | 526 |
| MQA (GQA, G=1) | 2 | 8 | 137 | 270 |
| MLA (dc=128, DRoPE16) | 10 | 17 | 160 | 311 |
| GVA + DRoPE16 | 2 | 9 | 152 | 302 |
| GVA (RoPE, no absorb) | 5 | 21 | 322 | 644 |
| GKA (ours, RoPE) | 7 | 15 | 166 | 328 |
| GKA + cross-group V | 11 | 19 | 171 | 332 |
| GKA + rotated V | 7 | 15 | 166 | 328 |
| GKA + cross-group + rotated V | 11 | 19 | 170 | 332 |
| Shared KV (K=V) | 3 | 11 | 162 | 324 |
