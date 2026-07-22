# Running large LLMs on old NVIDIA GPUs (Tesla V100 · RTX 2080 Ti)

Working recipes, hard-won gotchas and **real measured benchmarks** for serving large language models on pre-Ampere NVIDIA hardware — Volta (`sm_70`) and Turing (`sm_75`).

Most of the modern inference stack quietly assumes `sm_80+`. On V100 and 2080 Ti you hit a wall roughly every second step: quantized kernels refuse to load, PyTorch wheels ship without your arch, FlashAttention-2 needs Ampere, FP8 needs Hopper. Every recipe here was run on real hardware and the numbers below are our own measurements, not vendor claims.

> **Short version:** upstream vLLM will not serve AWQ/GPTQ on V100 or MoE-AWQ on 2080 Ti. You need architecture-specific forks. Which one, and with exactly which flags, is what this repo documents.

## Compatibility matrix

| | Tesla V100 (Volta, `sm_70`) | RTX 2080 Ti (Turing, `sm_75`) | RTX 4090 (Ada, `sm_89`) |
|---|---|---|---|
| Upstream vLLM + AWQ/GPTQ | ❌ `Minimum capability: 75` | ❌ MoE-AWQ dies in Marlin | ✅ |
| Working engine | [1Cat-vLLM fork](docs/volta-v100.md) | [weicj fork](docs/turing-2080ti.md) | upstream |
| Attention backend | `FLASH_ATTN_V100` | FlashQLA (fork) / Triton | FlashAttention-2 |
| dtype | `float16` only | `float16` only | bf16 |
| FlashAttention-2 | ❌ needs `sm_80` | ❌ | ✅ |
| FP8 / FP4 | ❌ needs Hopper/Ada | ❌ | FP8 ✅ |
| Real tensor parallelism | ✅ fork, or `llama.cpp -sm tensor` | ✅ fork | ✅ |
| Quantization that works | AWQ (W4A16) via fork | AWQ dense + AWQ MoE via fork | everything |

## Guides

- **[Tesla V100 (Volta, sm_70)](docs/volta-v100.md)** — what breaks and why; 1Cat-vLLM wheel install (incl. the `PackageNotFoundError` shim); building `llama.cpp` with real tensor parallelism and NCCL; Ollama configuration.
- **[RTX 2080 Ti (Turing, sm_75)](docs/turing-2080ti.md)** — building the weicj fork from source; `turboquant_k8v4` KV cache; the `awq` vs `awq_marlin` rule that changes throughput by **7.8×**.
- **[Benchmarks](docs/benchmarks.md)** — measured tokens/s across V100, 2080 Ti, A100 and 4090-48GB.
- **[Ubuntu 26.04 + CUDA](docs/ubuntu-2604-cuda.md)** — Python 3.14, gcc-15, cmake 4.2 and the glibc 2.41 `math_functions.h` patch that unblocks nvcc.
- **[GLM-5.2 on Volta: why it cannot work](docs/glm-5.2-on-volta.md)** — research writeup on DSA sparse attention and the only remaining path.
- **[Troubleshooting](docs/troubleshooting.md)** — garbled `Ġ`/`Ċ` output, the Ollama context trap that costs you 7× speed, and more.

## Headline numbers (our measurements)

| Hardware | Model | Config | Speed |
|---|---|---|---|
| 4× V100 32GB | DeepSeek-R1-70B-AWQ | 1Cat fork, TP=4 | **40 tok/s** single · 92 @16 parallel |
| 4× V100 32GB | Qwen3.5-122B-A10B-AWQ | 1Cat fork 1.2.0, TP=4 | **69.8 tok/s** single |
| 4× 2080 Ti 22GB | Qwen3.5-122B-A10B-AWQ | weicj fork, TP=4 | **52 tok/s** single |
| 4× 2080 Ti 22GB | Qwen3.6-27B-AWQ ×2 | 2 instances TP=2 + MTP K=3 | **152 tok/s** aggregate |
| 2× RTX 4090 48GB | Qwen3.6-35B-A3B | upstream | **149 tok/s** single |

Full tables, including the failures, in [docs/benchmarks.md](docs/benchmarks.md).

## Three lessons that save the most time

1. **Check your arch is actually in the wheel.** `python -c "import torch; print(torch.cuda.get_arch_list())"` must contain `sm_70`. Torch built against cu128/cu130 dropped Volta — you want `torch 2.5.1+cu124`, or a fork wheel that rebuilt it.
2. **A dense 70B cannot give you 60 tok/s on V100.** 900 GB/s × 4 ÷ 40 GB ≈ 90 tok/s is the hard memory-bandwidth ceiling. Benchmarks claiming 60 tok/s are MoE models (~3B active parameters/token). Memory bandwidth decides inference speed — NVLink matters far less than people assume.
3. **Every GPU generation needs its own fork.** Mixing them up produces crashes that look like model bugs. Turing → weicj, Volta → 1Cat, Ada → upstream.

## Contributing

Corrections and additional measurements are very welcome — especially if you have V100/2080 Ti results that contradict ours. Open an issue with your exact flags, engine version and `nvidia-smi` output.

## License

MIT — see [LICENSE](LICENSE).

---

Maintained by **[NextGen-PC](https://nextgen-pc.online/)** — we build and sell ready-made GPU servers for local LLM inference in Russia (Tesla V100 / A100, RTX 4090 48GB, RTX 2080 Ti 22GB). Everything documented here comes out of hardware we actually assemble, test and ship. Questions about a build are welcome in the issues.
