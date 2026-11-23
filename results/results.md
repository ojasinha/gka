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
| Shared KV (K=V) | 2048 | 2.0 | 8.0 | 128.0 | 256.0 |

| Method | decode tok/s B=1 T=1024 | decode tok/s B=1 T=4096 | decode tok/s B=16 T=4096 | decode tok/s B=32 T=4096 | prefill ms B=1 T=1024 | prefill ms B=1 T=4096 | prefill ms B=16 T=4096 | prefill ms B=32 T=4096 |
|---|---|---|---|---|---|---|---|---|
| GQA (baseline) | 95 | 94 | 1456 | 2924 | 23 | 93 | 566 | 1119 |
| MQA (GQA, G=1) | 94 | 94 | 1450 | 2985 | 23 | 92 | 579 | 1118 |
| MLA (dc=128, DRoPE16) | 79 | 78 | 1190 | 2423 | 27 | 110 | 668 | 1332 |
| GVA + DRoPE16 | 82 | 82 | 1236 | 2473 | 26 | 106 | 658 | 1304 |
| GVA (RoPE, no absorb) | 91 | 93 | 1156 | 1401 | 24 | 94 | 671 | 1567 |
| GKA (ours, RoPE) | 101 | 103 | 1578 | 2804 | 21 | 86 | 526 | 1069 |
| Shared KV (K=V) | 100 | 100 | 1543 | 2806 | 22 | 87 | 532 | 1070 |

| Method | decode peak MB B=1 T=1024 | decode peak MB B=1 T=4096 | decode peak MB B=16 T=4096 | decode peak MB B=32 T=4096 |
|---|---|---|---|---|
| GQA (baseline) | 13 | 16 | 264 | 526 |
| MQA (GQA, G=1) | 2 | 8 | 137 | 270 |
| MLA (dc=128, DRoPE16) | 10 | 17 | 160 | 311 |
| GVA + DRoPE16 | 2 | 9 | 152 | 302 |
| GVA (RoPE, no absorb) | 5 | 21 | 322 | 644 |
| GKA (ours, RoPE) | 7 | 15 | 166 | 328 |
| Shared KV (K=V) | 3 | 11 | 162 | 324 |
