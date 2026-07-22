# Troubleshooting

## Output full of `Ġ` and `Ċ` instead of spaces and newlines

Happens with AWQ repacks of Llama-family models (e.g. `DeepSeek-R1-Distill-Llama-70B-AWQ`).

**Cause:** `tokenizer_config.json` declares `"tokenizer_class": "LlamaTokenizer"` (the slow SentencePiece one), but the repack ships no `tokenizer.model` file — so the detokenizer never decodes byte-level BPE.

**Fix:** in the model directory, change

```json
"tokenizer_class": "PreTrainedTokenizerFast"
```

(the repack does include `tokenizer.json`), then restart the server.

**Verify:** `AutoTokenizer.from_pretrained(dir)` should report a `TokenizersBackend` class and decode non-ASCII text cleanly.

Worth checking on *every* AWQ model swap — it's a per-model packaging problem, not an engine bug.

---

## Model loads, then the first request returns `CUDA error: invalid argument`

On **Turing (`sm_75`)** with a quantized MoE model, this is the Marlin wall — there is no Turing MoE kernel. See [turing-2080ti.md](turing-2080ti.md). Check the log for `Using 'MARLIN' WNA16 MoE backend`.

On **Volta (`sm_70`)** with Qwen3.5/3.6 MoE, this is CUDA-graph capture failing in `custom_all_reduce.cuh`. Add `--disable-custom-all-reduce`.

---

## `no kernel image is available for execution on the device`

Your PyTorch build doesn't include your GPU's architecture.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

If `sm_70` (V100) or `sm_75` (2080 Ti) is missing, you have a cu128/cu130 wheel that dropped it. Use `torch 2.5.1+cu124`, or the rebuilt torch that ships with an architecture-specific fork.

---

## `ValueError: quantization method awq not supported. Minimum capability: 75. Current: 70`

Stock vLLM refusing AWQ on Volta. Expected — use the 1Cat fork. See [volta-v100.md](volta-v100.md).

---

## `NotImplementedError: Could not run '_C::awq_marlin_repack'`

An AWQ **MoE** model being routed into Marlin on a card that has no such kernel. On Turing use the weicj fork with `--quantization awq` (not `awq_marlin`). On Volta use the 1Cat fork.

---

## `device CUDA0 does not support split buffers` (llama.cpp)

The old `--split-mode row` was removed — the CUDA backend no longer exports `ggml_backend_split_buffer_type`. Use `--split-mode tensor` (add `-fit off`, since automatic fitting isn't implemented for tensor mode and will abort).

---

## `No available memory for cache blocks`

Weights ate the card. On 22 GB cards with a large MoE model, use the weicj fork's `--kv-cache-dtype turboquant_k8v4`; more generally lower `--max-model-len`, drop `--max-num-seqs`, try `--kv-cache-dtype fp8`, and raise `--gpu-memory-utilization` cautiously (0.96–0.97).

---

## Generation is ~2 tok/s and `nvidia-smi` shows the GPUs mostly idle

Classic Ollama context trap: with no explicit `num_ctx` it loads the model's maximum context, the KV cache doesn't fit, and it silently offloads to CPU. Send `{"options":{"num_ctx":8192}}`. Confirm with `ollama ps` — PROCESSOR should read 100% GPU.

---

## Worker dies with no traceback: `Failed core proc(s): {}`

Usually **system RAM**, not VRAM. Loading a 77 GB model on a box with 16 GB of RAM drives it into swap thrashing and the worker is killed. Add RAM or swap, or use a smaller model.

---

## Is my tensor parallelism real, or is it actually pipeline?

During generation, watch `nvidia-smi`:

- **Real TP** — all cards at similar utilization and power *at the same time*, VRAM split evenly.
- **Pipeline** — cards light up in sequence, one busy at a time.

For Ollama specifically, `journalctl -u ollama | grep "CUDA. .*layers"` showing `CUDA0: 11 layers` means layer-split, i.e. pipeline.

GGUF via llama.cpp in pipeline mode is capped at roughly the bandwidth of a *single* card, and concurrency does not scale — that's the trade-off against AWQ + real TP in a fork.
