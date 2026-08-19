# NVLink on V100 SXM2 carrier boards

Everything below was measured on 4× Tesla V100-SXM2-32GB kits mounted on third-party
carrier boards — the kind that shows up on the second-hand market. If you are buying
used SXM2 hardware, the NVLink mesh is the part most likely to be quietly broken, and
the part sellers almost never test.

## SXM2 is not a card

Tesla V100 shipped in two form factors. The **PCIe** version is a normal add-in board.
The **SXM2** version is a bare module: no slot connector, no power connector. It bolts
onto a carrier board (also called a baseboard or mezzanine board) which supplies power,
mounts the heatsink and — the point of this document — routes NVLink between modules.

Practical consequence: you cannot buy four SXM2 modules and plug them into a
motherboard. You need the carrier board, and its quality decides whether NVLink works.

## What a healthy 4-GPU mesh looks like

V100 SXM2 has **six NVLink 2.0 links** per GPU. Each link reports **25.781 GB/s**.
With four modules every GPU spends all six links on its three neighbours, so a healthy
kit shows **24 active links in total**.

```
nvidia-smi nvlink -s
```

Expected output: six `25.781 GB/s` lines under each of the four GPUs. Any line reading
`<inactive>` is a defect.

```
nvidia-smi topo -m
```

This shows how many links bond each pair — `NV1` (one link) or `NV2` (two links).
On a fully populated four-module board most pairs are `NV2`. A pair showing `NV1` where
the board routes two links means one of the two died: the pair still communicates, at
half the bandwidth.

Do not assume a symmetric mesh on larger boards. Eight V100 modules without an NVSwitch
are **two NV2 quads**, not one fully connected group — check `topo -m` before choosing
a tensor-parallel size that spans the boundary.

## A dead link is usually mechanical

This is the single most useful thing in this document. Across several used kits, links
reported as `<inactive>` were fixed **by re-torquing the module retention screws**, not
by replacing GPUs. It happened three times in a row on different kits.

SXM2 modules are pressed against a dense contact array with significant force. A screw
that is loose, or tightened unevenly so the module sits at a slight angle, produces
exactly this signature: the GPU is otherwise perfectly healthy, compute works, memory
works, but one or two links refuse to train.

Procedure:

1. Power the machine down completely and disconnect power from the carrier board.
2. Re-torque the retention screws on **all** modules, crosswise and evenly — not just
   the "suspect" one. A link belongs to a pair of GPUs.
3. Power up and re-run `nvidia-smi nvlink -s`.

Only if the link is still dead after a proper re-seat is the module itself suspect.

## Error counters lie about which card is at fault

NVLink CRC error counters increment **on both GPUs of a pair**. Seeing errors on GPU 2
does not mean GPU 2 is bad — the actual bad contact may be on GPU 0, the card on the
other end of that link. Always read the topology first and treat the pair as the unit
of diagnosis, not the individual GPU.

```
nvidia-smi nvlink -e          # error counters per link
```

## Synthetic checks pass on hardware that fails under load

A kit can show 24/24 links, zero ECC errors and a clean `dmesg`, and still fall over
once you actually push traffic through the mesh. The failure surfaces as **Xid 74** —
the NVLink error code — in the kernel log.

So the acceptance procedure has two halves, and the second one is not optional:

```
# static
nvidia-smi nvlink -s
nvidia-smi --query-gpu=index,serial,ecc.mode.current,\
  ecc.errors.uncorrected.aggregate.total --format=csv

# under load — see the gpu-burn note below
./gpu_burn 300

# then check nothing appeared
dmesg | grep -i "NVRM: Xid"
```

Relevant Xid codes when working with SXM2 kits:

| Xid | Meaning |
|---|---|
| 48 | Double-bit ECC error — the memory is failing, not a contact issue |
| 74 | NVLink error — the one that appears under mesh load |
| 79 | GPU fell off the bus — driver, power or seating |

## Two gotchas that waste an afternoon

**gpu-burn must be built for Volta.** The default build targets newer architectures and
dies on V100 with `no kernel image is available for execution on the device`, which
reads like a broken GPU:

```
make COMPUTE=70
```

**Do not wrap gpu-burn in `timeout`.** `timeout N ./gpu_burn` kills the parent before it
prints the per-GPU verdict, and you get exit code 124 with no result at all. The utility
already exits on its own duration argument — pass the duration and let it finish.

## PCIe width is not the bottleneck people expect

Carrier boards commonly reach the host through a PCIe switch (PLX PEX 8748 and
relatives), which leaves each GPU at **Gen3 x4**. That number alarms people. For LLM
inference it barely matters: PCIe carries weight loading and control traffic, while
tensor-parallel all-reduce traffic goes over NVLink, which is two orders of magnitude
faster. What you actually notice is a slower initial model load, not slower generation.

## Reference numbers from healthy kits

Measured on 4× V100-SXM2-32GB with a correctly seated mesh:

| Metric | Value |
|---|---|
| Active NVLink links | 24 / 24 at 25.781 GB/s |
| gpu-burn, FP32 per card | 13.9 – 14.5 TFLOPS (89–92% of the 15.7 TFLOPS spec peak) |
| Power draw per card under burn | 289 – 300 W (300 W cap) |
| GPU temperature under burn, adequate airflow | 41 – 54 °C |
| ECC errors after a 5-minute burn | 0 |

A spread wider than about 15 °C between cards in the same chassis is not normal
variance — it points at airflow or heatsink contact on the hot module.
