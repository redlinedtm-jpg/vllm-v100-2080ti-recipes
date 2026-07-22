# Benchmarks

All numbers below are our own measurements on hardware we assembled, using vLLM (or the noted engine) with an OpenAI-compatible endpoint. `single` = one stream; `@N` = N concurrent requests, aggregate throughput.

Treat them as order-of-magnitude guidance: exact numbers move with engine version, context length and prompt shape.

## 4× Tesla V100 SXM2 32GB (128 GB total, NVLink)

| Model | Engine / config | Single | Concurrent |
|---|---|---|---|
| DeepSeek-R1-70B-AWQ | 1Cat fork, TP=4, `FLASH_ATTN_V100` | **40 tok/s** | 92 @16 |
| DeepSeek-R1-70B Q4 | llama.cpp `-sm tensor` + NCCL | 17.6 tok/s | — |
| DeepSeek-R1-70B Q4 | Ollama (pipeline), ctx 8192 | 14.6 tok/s | ~30 @8 |
| Qwen3.5-122B-A10B-AWQ | 1Cat fork 1.2.0, TP=4 | **69.8 tok/s** | — |
| 120B | vLLM fork, TP=4 | 45 tok/s | 433 @16 |
| Qwen3.6-35B-A3B | vLLM, TP=4 | 70 tok/s | 841 @64 |
| Gemma-4-12B | llama.cpp TP, 4 cards | 44.6 tok/s | 277 @16 |

Note the spread on the *same* 70B model: **40 vs 17.6 vs 14.6 tok/s** depending purely on engine. Engine choice matters more than anything else you can tune here.

## 4× RTX 2080 Ti 22GB (88 GB total)

| Model | Engine / config | Single | Concurrent |
|---|---|---|---|
| Qwen3.5-122B-A10B-AWQ | weicj fork, TP=4, `turboquant_k8v4` | **52 tok/s** | — |
| Qwen3.6-27B-AWQ ×2 instances | 2× TP=2 (per NVLink pair), `awq_marlin`, **MTP K=3** | 76–83 per instance | **152.6 aggregate** |
| Qwen3.6-27B-AWQ ×2 instances | same, without MTP | 46 per instance | 90.4 aggregate |
| Llama-3.3-70B-AWQ | weicj fork, `awq_marlin` | 30.5 tok/s | 160 aggregate |
| Llama-3.3-70B-AWQ | weicj fork, plain `awq` ⚠️ wrong flag | 3.9 tok/s | — |
| Qwen3.6-35B-A3B vanilla fp16 | TP=4 | 22.8 tok/s | — |

## 4× Tesla A100 (128 GB total)

| Model | Config | Single | Concurrent |
|---|---|---|---|
| Qwen3.6-35B-A3B | upstream vLLM | 95 tok/s | 1826 @64 |
| 122B | upstream vLLM | 53.7 tok/s | 742 @32 |

Roughly **2× V100** on the same workload.

## 2× RTX 4090 48GB (96 GB total)

| Model | Config | Single | Concurrent |
|---|---|---|---|
| Qwen3.6-35B-A3B | upstream vLLM | **149 tok/s** | 550 @8 · 1171 @32 |
| 122B | upstream vLLM | 83 tok/s | — |

Two 48 GB 4090s beat both four-card server setups on single-stream speed. Newer architecture and much higher per-card bandwidth outrun the older cards' larger aggregate memory.

## The ceiling you cannot argue with

Inference speed on these cards is **memory-bandwidth bound**, so you can compute the hard limit before buying anything:

```
V100: 900 GB/s × 4 cards ÷ 40 GB (model size) ≈ 90 tok/s absolute ceiling
```

A dense 70B will therefore never give you 60 tok/s single-stream on V100 — physically impossible, regardless of engine. Published benchmarks showing "60 tok/s on V100" are **MoE models** (Qwen3-30B/35B-**A3B** and similar, ~3B active parameters per token), where only a fraction of weights is read per token.

Corollary: if you want single-stream speed on older cards, pick MoE architectures. If you want to fit the largest possible model, pick aggregate VRAM. And NVLink is far less decisive for inference than marketing suggests — bandwidth per card is what you feel.
