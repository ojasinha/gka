## Quality

Seeds per variant: gqa=1, gka=1, gka_drope=1, gva_drope=1, gva=1, sharedkv=1, gqa_g1=1, gka_g4=1  (mean ± std across seeds where >1)

| Method | Cache/token/layer | vs GQA | Params | Val loss | Val PPL | HellaSwag | ARC-E | ARC-C | OBQA | PIQA | WinoG | Avg (6) | LAMBADA acc | LAMBADA ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GQA (baseline) | 256 | - | 48.31M | 4.4109 | 82.34 | 26.0 | 33.2 | 21.8 | 24.2 | 53.6 | 51.1 | 34.98 | 3.3 | 18593 |
| GKA (ours, RoPE) | 128 | 50.0% smaller | 48.05M | 4.5149 | 91.37 | 25.6 | 32.2 | 21.5 | 24.0 | 53.4 | 49.4 | 34.35 | 1.2 | 39957 |
| GKA + DRoPE16 (ours) | 144 | 43.8% smaller | 48.64M | 4.4490 | 85.54 | 25.8 | 31.7 | 21.8 | 25.4 | 54.4 | 51.4 | 35.08 | 2.4 | 24760 |
| GVA + DRoPE16 | 144 | 43.8% smaller | 48.64M | 4.4450 | 85.20 | 26.1 | 31.5 | 20.6 | 24.4 | 53.8 | 51.5 | 34.67 | 2.2 | 23646 |
| GVA (RoPE, no absorb) | 128 | 50.0% smaller | 48.05M | 4.4989 | 89.92 | 25.7 | 33.1 | 22.3 | 25.4 | 53.3 | 49.8 | 34.92 | 1.2 | 34508 |
| Shared KV (K=V) | 128 | 50.0% smaller | 47.78M | 4.5361 | 93.33 | 25.7 | 32.2 | 22.6 | 24.8 | 53.0 | 48.9 | 34.53 | 1.3 | 38187 |
| MQA (GQA, G=1) | 128 | 50.0% smaller | 47.78M | 4.4385 | 84.65 | 25.8 | 32.5 | 21.5 | 26.0 | 54.0 | 49.5 | 34.88 | 2.7 | 23018 |
| GKA, G=4 (ours) | 256 | 0.0% smaller | 48.57M | 4.4754 | 87.83 | 25.7 | 32.3 | 21.9 | 25.0 | 53.4 | 50.4 | 34.79 | 1.7 | 29043 |

Accuracies in %. HellaSwag/ARC/OBQA/PIQA use length-normalised acc (acc_norm); WinoGrande uses partial scoring. Chance: hellaswag 25%, arc_easy 25%, arc_challenge 25%, openbookqa 25%, piqa 50%, winogrande 50%.

## Length extrapolation (trained at 512 tokens)

| Method | loss @512 | loss @1024 | loss @2048 | last-256-token loss @2048 |
|---|---|---|---|---|
| GQA (baseline) | 4.4646 | 4.8486 | 5.3765 | 5.9353 |
| GKA (ours, RoPE) | 4.5652 | 4.7291 | 5.0703 | 5.5099 |
| GKA + DRoPE16 (ours) | 4.5039 | 4.6407 | 5.5247 | 6.4802 |
| GVA + DRoPE16 | 4.4977 | 4.6414 | 5.5495 | 6.5080 |
| GVA (RoPE, no absorb) | 4.5500 | 4.7520 | 5.2421 | 5.8581 |
| Shared KV (K=V) | 4.5883 | 4.9947 | 5.4281 | 5.9613 |
| MQA (GQA, G=1) | 4.4947 | 4.7826 | 5.3970 | 6.0702 |
| GKA, G=4 (ours) | 4.5273 | 4.6194 | 5.0127 | 5.5545 |

## Inference (RTX 3050 6GB laptop, bf16, PyTorch eager, 64 decoded tokens)

| Method | KV bytes/token (all layers) | cache MB B=1 T=512 | cache MB B=1 T=2048 | cache MB B=16 T=2048 | cache MB B=32 T=4096 |
|---|---|---|---|---|---|
| GQA (baseline) | 4096 | 2.0 | 8.0 | 128.0 | 512.0 |
| GKA (ours, RoPE) | 2048 | 1.0 | 4.0 | 64.0 | 256.0 |
| GKA + DRoPE16 (ours) | 2304 | 1.1 | 4.5 | 72.0 | 288.0 |
| GVA + DRoPE16 | 2304 | 1.1 | 4.5 | 72.0 | 288.0 |
| GVA (RoPE, no absorb) | 2048 | 1.0 | 4.0 | 64.0 | 256.0 |
| Shared KV (K=V) | 2048 | 1.0 | 4.0 | 64.0 | 256.0 |
| MQA (GQA, G=1) | 2048 | 1.0 | 4.0 | 64.0 | 256.0 |
| GKA, G=4 (ours) | 4096 | 2.0 | 8.0 | 128.0 | 512.0 |

| Method | decode tok/s B=1 T=512 | decode tok/s B=1 T=2048 | decode tok/s B=16 T=2048 | decode tok/s B=32 T=4096 | prefill ms B=1 T=512 | prefill ms B=1 T=2048 | prefill ms B=16 T=2048 | prefill ms B=32 T=4096 |
|---|---|---|---|---|---|---|---|---|
| GQA (baseline) | 104 | 103 | 1598 | 3342 | 11 | 68 | 868 | 6061 |
| GKA (ours, RoPE) | 117 | 110 | 1908 | 1345 | 10 | 67 | 866 | 6396 |
| GKA + DRoPE16 (ours) | 94 | 93 | 1463 | 2502 | 12 | 82 | 977 | 6910 |
| GVA + DRoPE16 | 91 | 88 | 1392 | 2473 | 12 | 77 | 981 | 6931 |
| GVA (RoPE, no absorb) | 101 | 102 | 517 | 262 | 10 | 71 | 968 | 9376 |
| Shared KV (K=V) | 109 | 107 | 1981 | 1361 | 10 | 67 | 879 | 6492 |
| MQA (GQA, G=1) | 102 | 104 | 1612 | 3355 | 11 | 69 | 869 | 5976 |
| GKA, G=4 (ours) | 117 | 113 | 1323 | 738 | 10 | 69 | 889 | 6963 |

| Method | decode peak MB B=1 T=512 | decode peak MB B=1 T=2048 | decode peak MB B=16 T=2048 | decode peak MB B=32 T=4096 |
|---|---|---|---|---|
| GQA (baseline) | 10 | 8 | 133 | 525 |
| GKA (ours, RoPE) | 5 | 9 | 87 | 328 |
| GKA + DRoPE16 (ours) | 5 | 9 | 87 | 332 |
| GVA + DRoPE16 | 1 | 5 | 83 | 328 |
| GVA (RoPE, no absorb) | 3 | 10 | 163 | 644 |
| Shared KV (K=V) | 1 | 5 | 83 | 324 |
| MQA (GQA, G=1) | 1 | 4 | 70 | 270 |
| GKA, G=4 (ours) | 7 | 14 | 167 | 648 |
