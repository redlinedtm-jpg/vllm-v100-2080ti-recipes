# RTX 2080 Ti 22GB (Turing, `sm_75`)

Turing is the unluckiest middle ground: too new to fall into the `sm_70` Triton-AWQ path that Volta forks enable, too old for real Marlin kernels (`sm_80+`). Stock vLLM cannot serve a quantized MoE model here at all.

The way out is the **[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)** fork, which ships FlashQLA SM70/SM75 kernels. With it, a 122B MoE model runs at **52 tok/s** on four modded 22 GB cards.

## The wall: AWQ-MoE on `sm_75`

Quantized MoE (WNA16/AWQ) in vLLM only has these backends: `FLASHINFER_TRTLLM` (`sm_90+`), `MARLIN` and `BATCHED_MARLIN` (both `sm_80+`). **There is no Turing kernel.** Marlin advertises itself as supported and then dies at runtime.

- **Symptom:** the model loads, you see `Application startup complete`, and the first request returns `CUDA error: invalid argument` or an empty response. The log says `Using 'MARLIN' WNA16 MoE backend`.
- **Exact failure point:** `marlin_moe.py → moe_wna16_marlin_gemm` (`awq_marlin.py`) .
- Neither `VLLM_USE_TRITON_AWQ` nor changing the attention backend helps — the MoE GEMM path is independent of attention.
- On stock vLLM you may not even get that far: `NotImplementedError: Could not run '_C::awq_marlin_repack'` during `process_weights_after_loading`.

What *does* run on stock vLLM on Turing: **unquantized MoE** (goes through Triton) and **dense AWQ** models. Neither touches the Marlin MoE repack.

## Building the weicj fork

Takes about 40 minutes.

**1. CUDA 12.8 is mandatory.** CUDA 13.x breaks FlashQLA compilation (`error: need 'typename'` inside ATen). Install toolkit-only, leaving the driver alone:

```bash
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit \
   --toolkitpath=/usr/local/cuda-12.8 --no-opengl-libs
```
The fork's `build.sh` finds `/usr/local/cuda-12.8` on its own.

**2. Clone and build** with these environment variables:

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive /opt/vllm2080
cd /opt/vllm2080
HOME=/root ASSUME_YES=1 MAX_JOBS=4 NVCC_THREADS=2 \
  CUDA_HOME=/usr/local/cuda-12.8 ./build.sh
```

- `HOME` must be set explicitly or the script fails with `HOME: unbound variable`.
- `ASSUME_YES=1` — otherwise it waits for a TTY.
- `MAX_JOBS=4 NVCC_THREADS=2` keeps peak RAM sane on a 16 GB box.
- Run it under `systemd-run --unit=build --collect` if you're over SSH — the build outlives a dropped connection.

**3. Known build gotcha — the 820 MB torch download.** The script's pipe restarts the download from zero on a flaky link and can loop forever. Fetch the wheel manually with resume, then re-run `build.sh`:

```bash
curl -fL -C - --retry 8 -o torch.whl \
  https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl
.venv/bin/python -m pip install --no-deps torch.whl
./build.sh   # continues to BUILD OK
```

Resulting venv is Python 3.11 with torch 2.11.0+cu128.

## Serving a 122B MoE

```bash
/opt/vllm2080/.venv/bin/vllm serve /models/Qwen3.5-122B-A10B-AWQ \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --quantization awq \
  --kv-cache-dtype turboquant_k8v4 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --host 0.0.0.0 --port 8000
```

Why each flag matters:
- **`--kv-cache-dtype turboquant_k8v4`** — the fork's own KV quantization. Without it you get `No available memory for cache blocks`: weights alone take 19.3 GB of each 22 GB card.
- **`--quantization awq`** (not `awq_marlin`) for MoE — the Marlin MoE repack isn't built, you'd get `NotImplementedError`. Plain AWQ routes to Triton MoE, which works.
- Do **not** add `--enforce-eager`.
- First request after startup runs at ~3 tok/s — that's one-time JIT warmup, not a problem.

## ⚠️ The rule that changes throughput 7.8×

On this fork, quantization choice depends on model type — and getting it wrong is enormously expensive:

| Model type | Flag | Measured |
|---|---|---|
| **Dense** (e.g. Llama-3.3-70B-AWQ) | `--quantization awq_marlin` | **30.5 tok/s** single / 160 aggregate |
| Dense, wrong flag | `--quantization awq` | 3.9 tok/s — **7.8× slower** |
| **MoE** (e.g. 122B-A10B) | `--quantization awq` | 52 tok/s |
| MoE, wrong flag | `awq_marlin` | `NotImplementedError` |

The fork's patched `sm_75` Marlin is genuinely fast for dense models. It just has no MoE repack.

## Two instances instead of one, with MTP

The cards are NVLinked in pairs (GPU0↔1, GPU2↔3; PCIe between pairs). For mid-size models, running **two TP=2 instances — one per NVLink pair — beats one TP=4 instance**:

```bash
# instance A
CUDA_VISIBLE_DEVICES=0,1 vllm serve /models/qwen36-27b-awq --tensor-parallel-size 2 \
  --quantization awq_marlin --max-model-len 32768 --port 8001 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# instance B: CUDA_VISIBLE_DEVICES=2,3 ... --port 8002
```

Measured on Qwen3.6-27B-AWQ:
- without MTP: 46 tok/s per instance (90.4 aggregate)
- **with MTP K=3: 76–83 tok/s per instance, 152.6 aggregate** (both measured running simultaneously)

Note: use the ordinary KV cache here, not `turboquant` (that one is for very long context).

## Memory is tight — 88 GB total

A 122B AWQ model is ~77 GB, leaving almost nothing for KV and workspace. Useful levers: `--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.96`, `--max-num-seqs 1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and for VL models `--limit-mm-per-prompt '{"image":0,"video":0}'` so vision memory isn't reserved.

**Watch system RAM, not just VRAM.** On a box with only 16 GB of system RAM, loading a 77 GB model made the worker die with no traceback (`Failed core proc(s): {}`) and drove the machine into swap thrashing. That is a RAM limit, not a GPU limit — provision system RAM or swap accordingly.
