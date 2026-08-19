# CMP 170HX: what it actually is, measured

CMP 170HX is the only card NVIDIA ever built on **GA100** — the same die as the A100.
It was sold as a mining card with memory, compute and PCIe crippled in firmware rather
than in silicon, and in 2026 the community found a way to undo most of that. Since then
the second-hand market has filled with listings advertising "80GB VRAM" and "basically an
A100".

We unlocked one and measured it. This page is what the numbers actually say — including a
widely repeated claim that turns out to be false, a gotcha that is not in the tool's
documentation, and how to tell a 170HX from a real A100 before money changes hands.

## What is physically on the board

| Card | Die | Consumer sibling |
|---|---|---|
| CMP 30HX | TU116 | GTX 1660 |
| CMP 50HX | TU102 | RTX 2080 Ti |
| CMP 90HX | GA102 | RTX 3080/3090 |
| **CMP 170HX** | **GA100** | **Tesla A100** |

Only the 170HX is related to the A100; unlocking the others yields at best an incomplete
GeForce. The board carries full-size 16 GB HBM2e stacks — four of them on the "8 GB"
SKU — so the memory was always physically present, just not reported.

NVIDIA locked it at five levels: memory geometry registers, an FMA/IMLA compute throttle,
PCIe pinned to Gen1 in firmware, twelve of sixteen PCIe lanes left without their coupling
capacitors on the PCB, and a signed vBIOS. That last one is why the forum folklore of
"just flash an A100 vBIOS" does not work — the unlock is a driver-level exploit, not a
firmware swap.

## Measured after unlock

Our card, device ID `10de:20c2`, on a stock desktop chassis:

| | Stock | Unlocked |
|---|---|---|
| Reported memory | 8192 MiB | **65536 MiB** |
| PCIe | Gen1 x4 | Gen2 x4 |
| FP32 matmul | throttled | **12.2 TFLOPS** |
| FP16 matmul | — | **164 TFLOPS** |
| BF16 matmul | — | **174 TFLOPS** |
| Memory bandwidth (copy R+W) | — | **1263 GB/s** |
| Compute capability | — | 8.0 |

Serving a 70B model in AWQ on this single card: **27.2 tok/s** single stream, **207.2
tok/s** at 8 concurrent streams, 58.8 GB of memory in use.

### The tensor cores are not disabled

This is worth stating plainly, because the opposite is repeated all over the internet:
**tensor cores work.** 164 TFLOPS FP16 and 174 TFLOPS BF16 is roughly 85% of what an A100
delivers once you normalise for SM count. For vLLM, AWQ and FlashAttention this behaves
as a genuine `sm_80` device — which, coming from a V100 fleet where every second component
needs an architecture-specific fork, is the single most attractive property of this card.

### The memory is real

A `nvidia-smi` screenshot reading 65536 MiB proves nothing on its own — the interesting
question is whether the upper region is addressable and stable. We wrote 58 GiB of unique
patterns in 2 GiB chunks and read them back: **zero corrupted chunks, no address
aliasing**. The memory is genuinely there.

That is not the same as proving it is stable over hours under load, and with ECC
unavailable (below) there is nothing to catch silent corruption if it is not.

## What unlocking does not give you

- **ECC cannot be enabled.** The hardware has it; the GSP firmware will not bring it up on
  this card. 64 GB of HBM without ECC on multi-hour inference is a real risk, not a
  footnote.
- **There is no NVLink.** Tensor parallelism across several cards has to cross PCIe at
  Gen2 x4 — an all-reduce per layer over that link is hopeless. Pipeline parallelism is
  the only sensible multi-card mode.
- **PCIe Gen3/Gen4 is permanently out of reach** — it is disabled by an OTP fuse in
  silicon, not by firmware. The x16 width can be restored by soldering the missing
  coupling capacitors, but the link rate cannot.
- **It is not an A100.** The A100 40/80GB has 108 SMs. The 170HX has 56 or 70. Even fully
  unlocked, that is roughly 52–65% of A100 compute at comparable memory bandwidth.
- **The configuration is fragile.** It lives in a patched kernel module tied to one driver
  version, requires Secure Boot off, and any routine driver or kernel upgrade breaks it.

Our card is nominally the "8 GB" SKU, which the specifications describe as 56 SMs — but it
reports **70 SMs**. Do not assume SKU labels predict what you get.

## The gotcha that is not in the documentation

On Ubuntu the stock driver is installed through DKMS, which puts its modules in
`/lib/modules/$(uname -r)/updates/dkms/`. The patched modules land in
`/lib/modules/$(uname -r)/updates/cmpunlocker/`. Both directories sit under `updates/`
with equal priority, and `depmod` resolves to the **DKMS copy**.

The install looks completely successful. After the reboot the card reports 8192 MiB again
and `dmesg` contains zero `SEC2_DEBUG` lines. It is easy to lose an evening concluding the
exploit does not work on your card.

The fix is an explicit override:

```bash
printf 'override nvidia * updates/cmpunlocker\noverride nvidia-drm * updates/cmpunlocker\n' \
  > /etc/depmod.d/cmpunlocker.conf
depmod -a $(uname -r)
update-initramfs -u -k $(uname -r)
reboot
```

Verify it took effect: `modinfo -n nvidia` must point at
`updates/cmpunlocker/nvidia.ko`.

Two smaller notes from our run: the cold power-cycle some guides insist on was not needed —
an ordinary reboot sufficed — and the `-open` driver straight from the Ubuntu repository
worked. Note that this is the one case where you *want* the open flavour; for Volta cards
it is the exact opposite, see [the open kernel module trap](volta-driver-trap.md).

## Telling a 170HX from an A100 before you buy

The boards look alike, neither has display outputs, and listings are increasingly
optimistic. Run these checks on a **stock, unpatched driver** — the community patch only
works on a specially prepared kernel, so on any normal machine the card reports itself
honestly.

| Check | A100 | CMP 170HX |
|---|---|---|
| `lspci -nn \| grep -i nvidia` | `20b0` / `20b2` / `20b5` / `20f1` | `20c2` / `2082` |
| SM count (`nvidia-smi -q \| grep -i multiproc`) | **108** | 56 or 70 |
| `pcie.link.gen.max`, `pcie.link.width.max` | Gen4 x16 | Gen1/Gen2, x4 |
| `nvidia-smi -q -d ECC` | enabled | unavailable |
| `nvidia-smi nvlink -s` | links present | nothing |

The SM count is the most honest single indicator, and the PCIe generation cannot be faked:
it is a fuse plus twelve unpopulated capacitor positions on the PCB.

## Verdict

As a cheap single card to hold a large model with pipeline parallelism, and to get a
modern `sm_80` software stack without hunting for architecture forks, it is genuinely
interesting. As a production A100 replacement, or as something to sell to a customer, it
is not: no ECC, no NVLink, half the compute, and a software configuration that any package
upgrade can break.

If you are being offered a "cheap A100", run the table above first.

## Sources

The unlock is community work; we contributed measurements, not the exploit.

- Project documentation: <https://170th-street.gitbook.io/hx>
- Unlock tool: <https://github.com/amoghmunikote/cmpunlocker>
- Fork adding PCIe Gen2 and register documentation: <https://github.com/abobasixseven/unlock-cmp-170hx>
- Teardown and independent testing: <https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/>
- FMA throttle analysis and LLM measurements: <https://arxiv.org/abs/2505.03782>
