# Benchmarks

All numbers below are our own measurements on hardware we assembled, end-to-end through an OpenAI-compatible API. `single` = one stream; `@N` = N concurrent requests. Generation-side (decode) tokens/s unless labelled *prefill*.

Treat them as order-of-magnitude guidance: exact numbers move with engine version, context length and prompt shape.

---

## Qwen3.5-122B-A10B (AWQ 4-bit, MoE, 10B active) on 4× V100 32GB

Full concurrency curve. Engine: 1Cat-vLLM fork 1.2.0, TP=4, CUDA graphs, `FLASH_ATTN_V100`, float16, context 8192. Weights ~77 GB; KV cache held 520,564 tokens (up to 63 concurrent requests).

| Scenario | Total tok/s | Per request |
|---|---|---|
| 1 request | **67.4** | 67.4 |
| 8 concurrent | 134.7 | 16.8 |
| 16 concurrent | 393.3 | 24.6 |
| 32 concurrent | 628.3 | 19.6 |
| 48 concurrent | 861.0 | 17.9 |
| 64 concurrent | **1059.1** | 16.5 |
| **Prefill** (3.6k prompt) | **3767** | — |

**Power and thermals under full load:** 837 W across the four cards (~209 W/card), peak 47 °C. NVLink P2P measured at 48.5 GB/s.

The standout is prefill: 3767 tok/s means a 3.6k-token prompt is processed in about a second. On these cards, tensor parallelism over NVLink pays off far more for prompt processing than for decode.

## Qwen3.6-35B-A3B (MoE, 3B active) on 4× V100 32GB — by context size

| Context | Scenario | Total tok/s |
|---|---|---|
| 16384 | 1 request | **69.8** |
| 16384 | Prefill (~2.4k) | 2230 |
| 16384 | 16 concurrent | 244 |
| 16384 | 32 concurrent | 606 |
| 10000 | 32 concurrent | 409 |
| 10000 | 48 concurrent | 724 |
| 10000 | 64 concurrent | 841 |
| 60000 | 30 concurrent (short prompts) | 517.7 |

Single-stream speed (~70 tok/s) barely depends on context size. VRAM use ~30.5 GB/card.

### A note on vLLM's "Maximum concurrency" figure

At context 60000 vLLM reported `Maximum concurrency 19.89x`, but all 30 requests ran genuinely in parallel with no queueing — KV usage was only ~7,000 of 1.19M tokens.

That figure is **theoretical**: it assumes every request fills the entire declared context. KV cache is allocated on demand, so with short prompts you can far exceed it. The ~20 limit only bites if clients actually submit full-length ~60K contexts. Don't size your deployment off that number — measure with your real prompt distribution.

## Other models on 4× V100

| Model | Engine / config | Single | Concurrent |
|---|---|---|---|
| DeepSeek-R1-70B-AWQ | 1Cat fork, TP=4, `FLASH_ATTN_V100` | **40** | 92 @16 |
| DeepSeek-R1-70B Q4 | llama.cpp `-sm tensor` + NCCL | 17.6 | — |
| DeepSeek-R1-70B Q4 | Ollama (pipeline), ctx 8192 | 14.6 | ~30 @8 |
| Llama-3.3-70B-AWQ (dense) | 1Cat fork, TP=2 | works | — |
| Gemma-4-12B | llama.cpp TP, 4 cards (64 GB build) | 44.6 | 277 @16 |

Note the spread on the *same* 70B model: **40 vs 17.6 vs 14.6 tok/s** depending purely on engine. Engine choice dominates every other tunable here.

---

## 4× RTX 2080 Ti 22GB (88 GB total)

| Model | Engine / config | Single | Concurrent |
|---|---|---|---|
| Qwen3.5-122B-A10B-AWQ | weicj fork, TP=4, `turboquant_k8v4` | **52** | — |
| Qwen3.6-27B-AWQ ×2 instances | 2× TP=2 (one per NVLink pair), `awq_marlin`, **MTP K=3** | 76–83 each | **152.6 aggregate** |
| Qwen3.6-27B-AWQ ×2 instances | same, without MTP | 46 each | 90.4 aggregate |
| Llama-3.3-70B-AWQ | weicj fork, `awq_marlin` | 30.5 | 160 aggregate |
| Llama-3.3-70B-AWQ | weicj fork, plain `awq` ⚠️ wrong flag | 3.9 | — |
| Qwen3.6-35B-A3B vanilla fp16 | TP=4 | 22.8 | — |

---

## 4× Tesla A100 vs 4× Tesla V100 — same model, same harness

Qwen3.6-35B-A3B, context 10000, up to 64 concurrent, 200 tokens generated. V100: TP=4, fp16. A100: PP=4, bf16.

| Load | 4× V100 | 4× A100 | A100 advantage |
|---|---|---|---|
| 1 request | 69.8 | 95.0 | ×1.36 |
| 32 concurrent | 409 | 913 | ×2.23 |
| 48 concurrent | 724 | 1302 | ×1.80 |
| 64 concurrent | 841 | **1826** | ×2.17 |

A100 also measured 53.7 tok/s single / 742 @32 on a 122B model.

**Where A100 pulls further ahead than the ×2 above:**

| Workload | Gain | Why |
|---|---|---|
| int4 (AWQ/GPTQ) 70B+ | ×2–4 | A100 has Marlin int4 kernels; V100 has none |
| Long context / heavy prefill (RAG, 32K–256K) | ×2–4 | FlashAttention-2 requires `sm_80` |
| Dense models (Llama-70B, Qwen-72B) | ×2–2.5 | Tensor compute ×2.5 plus native bf16 |
| bf16-native models | noticeable | V100 must fall back to fp16 |

The single physical factor behind the consistent ~2×: **memory bandwidth**, ~1.5–2 TB/s on A100 against ~0.9 TB/s on V100. Token generation is bound by it almost everywhere.

⚠️ Caveat on our own A100 rig: it is connected over **PCIe x4 with no NVLink** (a board limitation), which handicaps it on collective-heavy tensor-parallel work — hence PP=4 rather than TP=4 in the comparison above. A properly NVLinked A100 setup should do better still on TP workloads.

## 2× RTX 4090 48GB (96 GB total)

| Model | Single | Concurrent |
|---|---|---|
| Qwen3.6-35B-A3B | **149** | 550 @8 · 1171 @32 |
| Qwen3.5-122B | 83 | — |

Two 48 GB 4090s beat both four-card server setups on single-stream speed — newer architecture and much higher per-card bandwidth outrun the older cards' larger aggregate memory.

---

## Topology matters: 8× V100 is not "twice a 4× V100"

On an 8× V100 SXM2 board without NVSwitch, the links form **two NV2 quads**: GPU0–3 fully meshed, GPU4–7 fully meshed, and only PCIe between the quads.

Practical consequence: **TP=4 inside one quad runs on full NVLink; TP=8 has to cross PCIe** and pays for it on every all-reduce. Unless a model genuinely needs more than 128 GB, you are usually better off running *two independent TP=4 instances* — one per quad — than one TP=8 instance. The same logic applies to 4× 2080 Ti, where cards are NVLinked in pairs: two TP=2 instances beat one TP=4 (see the MTP row above, 152 vs ~90 tok/s aggregate).

Check your own topology before choosing a parallelism strategy:
```bash
nvidia-smi topo -m        # NV# = NVLink between cards, PXB/NODE = PCIe hop
nvidia-smi nvlink -s      # per-link speed
```

## The ceiling you cannot argue with

Inference is memory-bandwidth bound, so the hard limit is computable before you buy anything:

```
V100: 900 GB/s × 4 cards ÷ 40 GB (model size) ≈ 90 tok/s absolute ceiling
```

A dense 70B will therefore never reach 60 tok/s single-stream on V100 — physically impossible, whatever the engine. Published "60 tok/s on V100" numbers are **MoE models** (Qwen3-30B/35B-**A3B** and similar, ~3B active parameters per token), where only a fraction of the weights is read per token.

Corollary: for single-stream speed on older cards pick MoE architectures; to fit the largest possible model pick aggregate VRAM. And NVLink matters far less for decode than marketing suggests — per-card bandwidth is what you feel, while NVLink shows up mainly in prefill and collectives.
