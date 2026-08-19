# Cooling passive Tesla cards, and reading their temperatures correctly

Datacenter GPUs ship with passive heatsinks: no fan on the card, by design. They assume
the chassis pushes air through them. Put one in an ordinary quiet desktop case and it will
throttle or shut down, and the failure looks like a broken GPU rather than a cooling
mistake.

This page covers what healthy thermals look like on V100-class hardware under sustained
load, the two temperature readings people routinely misinterpret, and how to tell an
airflow problem from a hardware problem.

## Two temperatures, not one

`nvidia-smi` shows GPU core temperature by default. On HBM2 cards there is a second sensor
that matters more:

```
nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,\
  power.draw,clocks.sm --format=csv
```

**Memory runs consistently hotter than the core** — in our measurements by roughly 7–10 °C.
Watching only `temperature.gpu` and concluding everything is fine is a common way to miss
a marginal setup.

Measured across a 60-minute full-load FP32 burn on 4× V100 at 300 W per card:

| Sensor | Range |
|---|---|
| GPU core | 36 – 63 °C |
| HBM2 memory | 54 – 69 °C |

A shorter 10-minute burn on a different chassis: core peaked at 70 °C, memory at 73 °C, no
throttling and no errors. Both are healthy pictures.

## The 90 °C that is not a temperature

This one costs people real worry. Running:

```
nvidia-smi -q -d TEMPERATURE
```

produces a field called **GPU Shutdown Temp: 90 C**. It is a firmware constant — the
threshold at which the card will cut itself off — **not a current reading**. Seeing it in
the output does not mean anything is at 90 °C.

The live values are `GPU Current Temp` and `Memory Current Temp`, or the query fields
above.

## What healthy sustained load looks like

From a 60-minute burn on a properly ventilated 4× V100 chassis:

| Metric | Value |
|---|---|
| Power per card | 300 W (at cap) |
| Total system GPU draw | ~1194 W |
| Core temperature, steady state | 61 – 63 °C |
| Memory temperature, peak | 69 – 70 °C |
| Tensor throughput over 24 min | 217 TFLOP/s |
| Throttling events | none |
| Idle core temperature, same chassis | 13 – 14 °C |

The important property is not the absolute number but that it is **steady**. A card that
climbs continuously through a long run has insufficient airflow even if it never reaches
the throttle point.

## The most useful diagnostic is the spread between cards

Cards in one chassis, running the same workload, should land within roughly 15 °C of each
other. A card sitting 15–20 °C above its neighbours is not "the hot slot" — it is one of:

- blocked or short-circuited airflow path for that position,
- poor heatsink contact or dried thermal interface material,
- a fan (chassis or shroud) that is not actually spinning.

We diagnosed exactly this on a card that ran 73 °C while its neighbours held 44–53 °C
under identical load. Everything else about it was healthy.

## Verify the fan is actually spinning

For any card that does have a fan — a consumer card driving the console, or an aftermarket
shroud on a Tesla — check the duty cycle under load, not just the temperature:

```
nvidia-smi --query-gpu=index,temperature.gpu,fan.speed,power.draw,clocks.sm \
  --format=csv --loop=10
```

A healthy fan ramps well before the card gets hot. We watched a card climb from 43 °C to
94 °C in about 140 seconds with `fan.speed` reporting **0%** the entire time: clocks
collapsed from 1911 MHz to 139 MHz, then the GPU dropped off the bus with `Xid 79`. The
cause was a fan connector that had not been reseated after maintenance. With the fan
working, the same card plateaued at 78–79 °C with the fan at 55% and never throttled.

The lesson generalises: **log the fan duty cycle and the clock alongside the temperature.**
A temperature graph alone cannot distinguish "cool because it is well cooled" from "cool
because it has already throttled to a crawl".

## Reading the throttle reasons

```
nvidia-smi --query-gpu=index,clocks_event_reasons.active --format=csv
```

The bit that matters is whether you are limited by power or by heat. Hitting the power cap
under a burn is normal and expected — the card is doing its job. Thermal throttling means
the cooling is inadequate for the workload.

## Power capping as a cooling strategy

If a chassis cannot carry four cards at full draw, capping is a legitimate answer and
costs less performance than people expect on memory-bound inference workloads:

```
nvidia-smi -pl 250          # per-card limit, watts
```

Make it persistent across reboots with a small systemd unit; a cap set by hand disappears
on the next boot, and the first hot day afterwards is when you find out.

## Practical checklist

- Forced airflow through the card is mandatory, not optional.
- Watch **memory** temperature, not just core — it runs 7–10 °C higher.
- `GPU Shutdown Temp: 90 C` in `-q -d TEMPERATURE` is a constant, not a reading.
- Steady beats low: a plateau is healthy, a continuous climb is not.
- Compare cards against each other; a 15–20 °C outlier is a fault, not variance.
- Log fan duty and SM clock next to temperature, or you will misread a throttled card as
  a cool one.
- If the chassis cannot feed four cards, cap power rather than pretending it can.
