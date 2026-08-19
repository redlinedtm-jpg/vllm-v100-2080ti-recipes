# nvidia-smi finds no devices on V100 — the open kernel module trap

Symptom: four Tesla V100s are visible in `lspci`, the machine boots fine, DKMS reports a
module built for the running kernel — and `nvidia-smi` says it cannot communicate with
the driver. The kernel log is full of alarming lines about GPUs falling off the bus.

Nothing is wrong with the hardware. You installed a driver package that does not support
Volta.

## Why it happens

NVIDIA ships two flavours of the kernel driver: the traditional proprietary modules, and
the **open** kernel modules (packages with an `-open` suffix, and the default for several
recent branches). The open modules require a GSP-capable GPU, which means **Turing
(`sm_75`) and newer**. Volta (`sm_70`) is not supported and never will be.

If the machine previously ran newer cards — or you simply took the branch your
distribution recommended — you can end up with an `-open` driver on a V100 box and lose
an afternoon to what looks like four dead GPUs.

## The signature in the logs

```
NVRM: GPU 0000:xx:00.0: GPU has fallen off the bus.
NVRM: BAR0 is 0M @ 0x0 (PCI:0000:xx:00.0)
nvidia 0000:xx:00.0: probe with driver nvidia failed with error -1
nvidia 0000:xx:00.0: Unable to change power state from D3cold to D0, device inaccessible
NVRM: The NVIDIA probe routine failed for 4 device(s).
```

Two details separate this from real hardware failure:

- **All** GPUs fail identically. Genuine hardware faults are rarely so uniform.
- `BAR0 is 0M` means the driver never mapped the aperture at all — it did not get far
  enough to talk to the GPU, as opposed to talking to it and getting garbage.

Check what you actually have installed:

```
dpkg -l | grep -i nvidia | grep -- -open      # any hit here is the problem on Volta
```

## The fix

Replace the open driver with a proprietary **server** branch. In our case the
`580-server` branch worked; the important part is proprietary, not the exact number.

```
# 1. remove every package from the open branch
apt-get purge -y $(dpkg -l | awk '/nvidia.*610/{print $2}')

# 2. install the proprietary server driver
apt-get install -y --no-install-recommends nvidia-driver-580-server
```

Then reboot. Do not skip the reboot: once the GPUs have been left in D3cold with no
usable BAR mapping, a PCI rescan will not recover them.

## Three traps inside the fix

**Stale modules block the new DKMS build.** `apt purge` does not delete the compiled
modules from the kernel tree, and DKMS then refuses to install over them:

```
Error! Module version 580.173.02 for nvidia.ko.zst
is not newer than what is already found in kernel <version> (610.43.02).
```

Remove the leftovers by hand and force the install:

```
rm -f /lib/modules/$(uname -r)/updates/dkms/nvidia*.ko.zst
dkms install nvidia-srv/580.173.02 -k $(uname -r) --force
depmod -a $(uname -r)
```

Note the DKMS module for the server branch is named `nvidia-srv`, not `nvidia`.

**Leftover depmod overrides.** If the box was previously used with patched or unlocked
cards, `/etc/depmod.d/` may contain an override forcing a specific module path:

```
/etc/depmod.d/<something>.conf:override nvidia * updates/<something>
```

With that file present, `modprobe nvidia` keeps loading a module that no longer exists
or does not match. Move it out of the way before rebuilding.

**Never PCI-remove the display GPU.** If the box has a consumer card driving the console,
do not try `echo 1 > /sys/bus/pci/devices/<addr>/remove` followed by a rescan on it. That
card owns an active framebuffer; yanking it on a live system can take the machine down.
Reboot instead. (Learned the hard way.)

## Verifying the fix

```
cat /proc/driver/nvidia/version
nvidia-smi --query-gpu=index,name,serial,persistence_mode --format=csv
```

You should see all cards, in `P0`, with sane idle temperatures. If `lspci` shows the GPUs
but `nvidia-smi` still does not, and you have confirmed a proprietary driver is loaded,
only then start suspecting power delivery or seating.

## Which branch to use

Volta needs a driver branch that still carries `sm_70` support in both the kernel module
and the userspace libraries. Practical rule: pick the **`-server` proprietary** package
your distribution offers for datacenter GPUs, and pin it, because an unattended upgrade
that pulls in an open branch will silently reproduce this whole failure after the next
reboot.

Pin narrowly, by exact package name. A broad hold pattern such as `libnvidia-*` also
catches `libnvidia-ml-dev` and then the CUDA toolkit refuses to install with held broken
packages.
