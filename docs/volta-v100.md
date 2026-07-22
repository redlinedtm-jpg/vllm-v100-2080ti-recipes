# Tesla V100 (Volta, `sm_70`)

Volta is compute capability 7.0 — old enough that half the modern inference stack silently excludes it. This is the complete map of what works, what doesn't, and the exact commands.

## What does NOT work

| Thing | Failure | Why |
|---|---|---|
| Upstream vLLM + AWQ/GPTQ | `ValueError: quantization method awq not supported. Minimum capability: 75. Current: 70` | Quantized kernels in stock vLLM require `sm_75+` |
| torch with cu128 / cu130 (2.11+) | `no kernel image is available for execution on the device` | These builds dropped `sm_70`. Verify with `torch.cuda.get_arch_list()` |
| FlashAttention-2 | import/runtime error | Requires `sm_80` |
| FP8 (and bf16) | unsupported | Volta has no FP8 blocks; use `float16` |
| CUDA 13 toolkit | build failures | Volta support was removed in CUDA 13.0 — build with 12.x |

**Always check first:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# must contain 'sm_70'
```
`torch 2.5.1+cu124` has it. Fork wheels ship a rebuilt torch that also has it.

## What works

- **1Cat-vLLM fork** — the fastest option. TP=4 + AWQ + a native `FLASH_ATTN_V100` kernel. Roughly **2.3× faster than llama.cpp** on the same model.
- **llama.cpp with `-sm tensor`** — genuine tensor parallelism (weights *and* KV split across cards, all-reduce every token). Requires building yourself.
- **Ollama** — works out of the box, but it is *layer-split (pipeline)*, not tensor parallel. Fine for single-user chat.

---

## Recipe A — 1Cat-vLLM (fastest; prebuilt wheel, no compilation)

```bash
# 1. Python 3.12 venv (wheels are cp312). Ubuntu 26.04 ships 3.14, so use uv.
uv venv --python 3.12 /opt/1cat-venv

# 2. Install the wheel with BOTH indexes:
#    flashinfer-python only exists on PyPI, torch on the cu128 index.
WHL=1cat_vllm-1.2.1-cp312-cp312-linux_x86_64.whl
curl -sL https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.2.1/$WHL -o /tmp/$WHL
uv pip install --python /opt/1cat-venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 /tmp/$WHL
```

### Gotcha: `PackageNotFoundError: vllm`

The wheel installs itself as `1cat_vllm`, but the CLI calls `importlib.metadata.version("vllm")`. Fix with shim metadata:

```bash
SP=/opt/1cat-venv/lib/python3.12/site-packages
mkdir -p $SP/vllm-1.2.1.dist-info
printf "Metadata-Version: 2.1\nName: vllm\nVersion: 1.2.1\n" > $SP/vllm-1.2.1.dist-info/METADATA
```

### Serving

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
/opt/1cat-venv/bin/vllm serve /models/<AWQ-model> \
  --served-model-name my-model \
  --quantization awq \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --host 0.0.0.0 --port 8000
```

`CUDA_DEVICE_ORDER=PCI_BUS_ID` plus an explicit device list matters if the box has a weak display GPU alongside the V100s — otherwise the tensor-parallel split trips over it.

Result is an OpenAI-compatible API on `:8000/v1`. Any client that speaks "OpenAI-compatible" (Open WebUI, Chatbox, …) connects with `model id = served-model-name`.

### Qwen3.5 / Qwen3.6 MoE models (GDN linear attention)

Architectures like `qwen3_5_moe` (35B-A3B, 122B-A10B) need extra care on Volta:

- Use **1Cat-vLLM 1.2.0+** — 1.1.0 does not load the MoE experts.
- **`--disable-custom-all-reduce` is mandatory**, otherwise it crashes during CUDA-graph capture with `custom_all_reduce.cuh 'invalid argument'`.
- Requires torch 2.10.0+cu128 and `FLASH_ATTN_V100`.
- Do **not** pass `--swap-space` (the argument is rejected by this build).

Verified on 8× V100 (2026-07). Measured 69.8 tok/s single-stream on 4× V100.

---

## Recipe B — llama.cpp with real tensor parallelism

Useful when you need GGUF rather than AWQ. Slower than the fork, but simpler to trust.

> ⚠️ The old `--split-mode row` is **dead** in recent llama.cpp — the CUDA backend no longer exports `ggml_backend_split_buffer_type`, so you get `device CUDA0 does not support split buffers`. Use `-sm tensor`.

```bash
# deps
apt install -y cmake build-essential git gcc-13 g++-13 libcurl4-openssl-dev

# CUDA 12.9 toolkit via conda (see ubuntu-2604-cuda.md for why 12.9 and not 12.6)
conda create -y -n cuda129 -c nvidia/label/cuda-12.9.1 cuda-toolkit
CUDA=/opt/conda/envs/cuda129

# NCCL — without it, all-reduce is so slow that TP loses to pipeline
conda install -y -n cuda129 -c conda-forge nccl

export PATH=$CUDA/bin:$PATH CUDA_HOME=$CUDA
export LD_LIBRARY_PATH=$CUDA/lib:$CUDA/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp

cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCUDAToolkit_ROOT=$CUDA -DNCCL_ROOT=$CUDA \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --config Release -j$(nproc)
```

On Ubuntu 26.04 you will also need the `math_functions.h` patch before this compiles — see [ubuntu-2604-cuda.md](ubuntu-2604-cuda.md).

### Serving

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH=$CUDA/lib:$CUDA/targets/x86_64-linux/lib \
build/bin/llama-server -m model.gguf \
  -ngl 999 --split-mode tensor --tensor-split 1,1,1,1 \
  -fit off -np 2 -cb -c 8192 --host 0.0.0.0 --port 8000
```

`-fit off` is required: automatic parameter fitting is not implemented for tensor mode and aborts.

**Verifying you actually got TP, not pipeline:** during generation `nvidia-smi` should show *all four* cards at similar utilization and power *simultaneously*, with VRAM split evenly (e.g. ~10.3 GB per card for a 40 GB model). Pipeline mode lights the cards up one after another.

---

## Recipe C — Ollama

Works out of the box (it bundles llama.cpp built for `sm_70`), but it is pipeline-parallel, not TP.

`/etc/systemd/system/ollama.service.d/override.conf`:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_SCHED_SPREAD=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3"
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
```

**The context trap — this one costs 7× speed.** Without an explicit `num_ctx`, Ollama loads the model's *maximum* context (e.g. 131072), the KV cache balloons to ~160 GB, doesn't fit, and silently spills to CPU — you get ~2 tok/s. Always send:

```json
{"options": {"num_ctx": 8192}}
```

With that, a 70B Q4 runs fully on GPU at ~14.6 tok/s single / ~30 tok/s across 8 parallel requests.

Check with `ollama ps` (PROCESSOR should read 100% GPU). If `journalctl -u ollama | grep "CUDA. .*layers"` shows lines like "CUDA0: 11 layers", you are on pipeline, not TP.

### Reusing models Ollama already pulled

GGUF files live as blobs at `/usr/share/ollama/.ollama/models/blobs/sha256-<digest>`. The largest blob is the GGUF, and `llama-server -m <blob>` loads it directly — no need to re-download. The digest is in the manifest under `.../manifests/registry.ollama.ai/library/<model>/<tag>` (layer mediaType ending in `...model`).
