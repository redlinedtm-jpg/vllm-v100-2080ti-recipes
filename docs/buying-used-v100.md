# Acceptance checklist for used Tesla V100 SXM2 kits

Second-hand V100 modules are the cheapest way to get a lot of HBM2 into one box, and the
market is full of kits pulled from datacenters, mining farms and dead servers. Most of
them are fine. The ones that are not fail in ways that a five-minute check will not
reveal, and that no seller will volunteer.

This is the procedure we run on every kit before it goes anywhere near a customer.

## 1. Serial numbers must be unique

```
nvidia-smi --query-gpu=index,serial,uuid,vbios_version,inforom.img --format=csv
```

Four distinct serials, four distinct UUIDs. **Identical serials across modules, or a
single inforom image reported for the whole kit, is a hard stop** — it means the
identity data has been rewritten, and you have no idea what the modules actually are.

Note the serials down. They are also how you prove later which physical module a fault
belonged to, after cards have been moved between chassis.

## 2. ECC must be on and clean

```
nvidia-smi --query-gpu=index,ecc.mode.current,\
  ecc.errors.corrected.volatile.total,\
  ecc.errors.uncorrected.volatile.total,\
  ecc.errors.uncorrected.aggregate.total --format=csv
```

ECC enabled, and the **aggregate** uncorrected counter at zero. Aggregate survives
reboots, so it is the one a seller cannot reset by power-cycling before the demo.
A card that throws double-bit errors (`Xid 48`) under load has failing HBM and is not
repairable by reseating.

## 3. NVLink: all 24 links

```
nvidia-smi nvlink -s
nvidia-smi topo -m
```

Six links per GPU at 25.781 GB/s, 24 total on a four-module kit, no `<inactive>` lines.
A dead link is usually a loose retention screw rather than a dead module — the full
diagnosis is in [NVLink on V100 SXM2 carrier boards](nvlink-sxm2.md).

## 4. Load test — the half everybody skips

Static checks pass on hardware that dies under load. Run a real burn across all cards
simultaneously.

```
# Volta needs its own build target, otherwise:
# "no kernel image is available for execution on the device"
make COMPUTE=70
./gpu_burn 300
```

Two practical notes:

- **Do not wrap it in `timeout`.** The wrapper kills the parent before the per-GPU
  verdict is printed and you get exit code 124 with no result. Pass the duration to
  `gpu_burn` itself.
- Run **all four at once**, not one at a time. Power delivery and airflow problems only
  appear at full load, and that is exactly what you are testing for.

Expected on a healthy kit:

| Metric | Healthy value |
|---|---|
| gpu-burn result | `OK` on every GPU, zero computation errors |
| FP32 throughput | 13.9 – 14.5 TFLOPS per card (89–92% of spec peak) |
| Power draw | 289 – 300 W per card against the 300 W cap |
| Temperature | 41 – 54 °C with adequate airflow |
| Temperature spread between cards | under ~15 °C |

A card sitting 15–20 °C above its neighbours is not "just the hot slot" — it is airflow
or heatsink contact, and it will throttle in production.

## 5. Kernel log after the burn

```
dmesg | grep -i "NVRM: Xid"
```

Must be empty. Codes worth recognising:

| Xid | Meaning | Verdict |
|---|---|---|
| 48 | Double-bit ECC | Failing memory — reject |
| 74 | NVLink error | Mesh problem; re-seat, retest, reject if it persists |
| 79 | GPU fell off the bus | Driver, power or seating — investigate before blaming the card |

We have seen kits pass every static check and then throw `Xid 74` under sustained load.
That is precisely why step 4 exists.

## What the kit does not include

Buying the modules is roughly half the job. Also required:

- **Carrier board** — SXM2 modules do not fit any slot; the board provides power, mounting
  and the NVLink routing.
- **Power** — four modules at 300 W is 1200 W for GPUs alone. Budget from 1600 W for the
  system. A machine that runs fine idle and drops out one minute into a full-load run is
  almost always power.
- **Airflow** — Tesla heatsinks are passive and assume forced chassis airflow. A quiet
  desktop case will cook them.
- **System RAM** — models are staged in host memory before being distributed to the GPUs.
  Aim for at least as much RAM as total VRAM; 128 GB of VRAM behind 32 GB of RAM is a
  bottleneck you will feel on every load.
- **The right driver** — a modern *open* kernel module will not drive Volta at all. See
  [the open kernel module trap](volta-driver-trap.md).

## Minimal acceptance script

```
set -e
nvidia-smi --query-gpu=index,serial,ecc.mode.current,\
  ecc.errors.uncorrected.aggregate.total --format=csv
nvidia-smi nvlink -s | grep -c "25.781 GB/s"      # expect 24
XID_BEFORE=$(dmesg | grep -c "NVRM: Xid" || true)
cd gpu-burn && ./gpu_burn 300
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,\
  ecc.errors.uncorrected.volatile.total --format=csv
XID_AFTER=$(dmesg | grep -c "NVRM: Xid" || true)
echo "Xid before=$XID_BEFORE after=$XID_AFTER"
```

If `XID_AFTER` is greater than `XID_BEFORE`, the kit failed regardless of how good the
static numbers looked.
