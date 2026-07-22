# GLM-5.2 on Volta: why tensor parallelism cannot work

Research writeup, verified 2026-07. Short answer: **you cannot serve GLM-5.2 with tensor parallelism on V100 (`sm_70`) via vLLM or SGLang — and W4A16 quantization does not rescue it.** The only working path on Volta is GGUF pipeline inference through llama.cpp.

This is worth writing down because the intuitive reasoning ("quantize it to int4 and it'll fit") is wrong for a non-obvious reason.

## Why quantization doesn't help

GLM-5.2 (`GlmMoeDsaForCausalLM`, ~745B total / ~40B active) uses **DSA — DeepSeek Sparse Attention** — plus MTP.

The DSA sparse-MLA kernels are tied to `sm_90`/`sm_100` (Hopper/Blackwell). It already fails on **A100 (`sm_80`)**, per the open vLLM issue [#35021](https://github.com/vllm-project/vllm/issues/35021), which identifies three separate layers of incompatibility:

1. `dsv3_fused_a_gemm` is `sm_90+` with no guard,
2. there is no sparse-MLA backend for `sm_80`,
3. the indexer hardcodes DeepGEMM `fp8_mqa_logits`.

PR #35271 fixes only the first layer and is explicitly "not sufficient for sm80". Volta is a further generation back, with no bf16 or FP8 at all.

**And W4A16 quantizes only the MoE experts.** Attention (MLA plus the DSA indexer) stays in BF16 and still calls `sm_90` FP8 kernels — so the blocker survives quantization completely intact. Model cards for W4A16 builds cite 8× B200, not older hardware.

Official recipes ([vLLM](https://recipes.vllm.ai/zai-org/GLM-5.2), [SGLang](https://lmsysorg.mintlify.app/cookbook/autoregressive/GLM/GLM-5.2)) state plainly: no Ampere or Volta fallback. The best community port, [renning22/glm-5.2-4090](https://github.com/renning22/glm-5.2-4090), bottoms out at **Ada `sm_89`** because it needs native FP8.

## The one path that does work: GGUF via llama.cpp

llama.cpp runs it because **it never executes DSA at all** — it loads the GLM DSA indexer tensors as optional and computes dense DeepSeek-V2-style MLA instead (convert PR #19460, loader PR #24770; the actual DSA runtime is draft #21149 and unmerged).

`0xSero/GLM-5.2-REAP-504B-GGUF` **Q4_K_XL (~325 GB)** loads on mainline llama.cpp without patches. Smaller: Q3_K_XL 259 GB, Q2_K_XL 111 GB. The indexer is approximated (duplicated from neighbouring layers, not bit-exact).

The price you pay:
- dense attention instead of sparse → expensive long-context prefill,
- **single-stream pipeline, not TP** — concurrency does not scale,
- ~10–20 tok/s (no V100 measurement yet; M3 Ultra lands around 19–23).

At 325 GB you need ~384 GB of VRAM (twelve 32 GB cards). On 8× V100 = 256 GB it does not fit — only Q2_K_XL (111 GB), or Q3 with CPU offload. Worth trying [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), which has better MoE/MLA CPU-GPU splitting and `-ot` expert offload.

If you need genuinely sparse GLM-5.2 with real throughput, the hardware floor is Ada `sm_89` / Hopper / Blackwell. The TP fallback on older NVIDIA is the earlier **GLM-4.6 / 4.7 W4A16 AWQ** (no DSA).

## Builds that are dead on Volta

| Build | Format | Why |
|---|---|---|
| `0xSero/GLM-5.2-504B` / `-Nvidia` | NVFP4 | Blackwell only |
| `PhalaCloud/…W4AFP8` | FP8 | no FP8 on Volta |
| `QuantTrio/GLM-5-AWQ` | AWQ | pulls FlashInfer, no `sm_70` |
| `0xSero/GLM-5.2-504B-W4A16` | W4A16 | DSA blocker remains (see above) |

## Sizes of other large MoE models (for capacity planning)

Assuming a 384 GB budget (12× V100 32GB):

| Model | Original | Best int4 for TP | Fits 384 GB? |
|---|---|---|---|
| Qwen3-235B-A22B-2507 | 470 GB | `QuantTrio/…-AWQ` **124 GB** | ✅ fastest, lots of room for KV |
| MiniMax-M3 | 854 GB | `…-AWQ-int4` **241 GB** | ✅ |
| Qwen3-Coder-480B-A35B | 960 GB (FP8 482) | `QuantTrio/…-AWQ` **252 GB** | ✅ |
| Ring-2.6-1T | 1042 GB FP8-native | no int4/GGUF exists | ❌ unrealistic on Volta |

## General lesson

On pre-Ampere hardware, **check the attention implementation before the parameter count**. Model size is a memory problem and quantization solves memory problems. A hardcoded `sm_90` attention kernel is an architecture problem, and no amount of quantization touches it.
