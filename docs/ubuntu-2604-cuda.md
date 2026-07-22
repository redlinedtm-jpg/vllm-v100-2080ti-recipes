# Ubuntu 26.04 + CUDA on old GPUs

Ubuntu 26.04 is a rough host for legacy-GPU inference work. Five things bite, in this order.

## 1. Python 3.14 is too new

26.04 ships Python 3.14, and there are no vLLM wheels for it (you need 3.11–3.12). Don't fight the system Python — get a standalone one:

```bash
uv venv --python 3.12 /opt/myenv   # uv downloads a standalone 3.12 itself
```
miniforge/conda works too.

## 2. gcc-15 is too new for nvcc

```bash
apt install -y gcc-13 g++-13
# then pass to cmake:
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
-DCMAKE_C_COMPILER=/usr/bin/gcc-13
-DCMAKE_CXX_COMPILER=/usr/bin/g++-13
```

## 3. No CUDA apt repo for 26.04 — and pick 12.9, not 12.6

Install the toolkit through conda instead:

```bash
conda create -y -n cuda129 -c nvidia/label/cuda-12.9.1 cuda-toolkit
```

**Why 12.9 specifically:** glibc 2.41 in 26.04 conflicts with the `math_functions.h` header of older CUDA releases —

```
error: exception specification is incompatible ... cospi/sinpi/rsqrt
```

That was fixed in CUDA ≥ 12.8. And you must stay on 12.x because **Volta support was removed in CUDA 13.0** — 12.9 is the newest release that still compiles `sm_70`.

## 4. The `math_functions.h` patch (needed even on 12.9)

Even CUDA 12.9 does not fully fix the glibc 2.41 clash, so nvcc still fails. Patch the header — this is the single most time-consuming thing to rediscover:

```bash
H=$CUDA/targets/x86_64-linux/include/crt/math_functions.h
cp -n $H ${H}.bak
sed -i -E 's/(__device_builtin__[[:space:]]+(double|float)[[:space:]]+(rsqrt|rsqrtf|sinpi|sinpif|cospi|cospif)\((double|float) x\));/\1 noexcept (true);/' $H
```

It adds the missing `noexcept (true)` to the six declarations that glibc 2.41 declares differently.

## 5. cmake 4.2 rejects old policy minimums

llama.cpp (and others) declare a `cmake_minimum_required` that 4.2 refuses:

```bash
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

Also export the conda CUDA paths *before* invoking cmake, or its CUDA-compiler test fails because it cannot find `libcudart`:

```bash
export PATH=$CUDA/bin:$PATH CUDA_HOME=$CUDA
export LD_LIBRARY_PATH=$CUDA/lib:$CUDA/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## Bonus: don't let pip resolve vLLM

`pip install vllm` can backtrack for *hours*. Use uv, and pin:

```bash
uv pip install vllm==0.6.6.post1   # pulls torch 2.5.1+cu124, which still has sm_70
```

## Bonus: long builds over SSH

Wrap anything long in a transient systemd unit so it survives a dropped connection:

```bash
systemd-run --unit=mybuild --collect bash /root/build.sh
journalctl -u mybuild -f
```

## Bonus: display flicker / black screen with a consumer card alongside

If the box has a cheap display GPU next to the compute cards, NVIDIA + Wayland can blank the monitor on idle and never wake it. On 26.04 there is **no Xorg session at all** (`/usr/share/xsessions/` doesn't exist), so `WaylandEnable=false` achieves nothing — there's no X session to fall back to. Remove the trigger instead:

```bash
systemctl restart gdm     # repaint now (a black screen is usually stuck DPMS)

U=unix:path=/run/user/1000/bus
sudo -u <user> DBUS_SESSION_BUS_ADDRESS=$U gsettings set org.gnome.desktop.session idle-delay 0
sudo -u <user> DBUS_SESSION_BUS_ADDRESS=$U gsettings set org.gnome.desktop.screensaver idle-activation-enabled false
sudo -u <user> DBUS_SESSION_BUS_ADDRESS=$U gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type nothing
systemctl mask sleep.target suspend.target
```

Diagnosis that it's DPMS and not a crash: `nvidia-smi` alive, `cat /sys/class/drm/card*/card*-HDMI*/status` = `connected`, session `Active=yes`. Compute is unaffected either way.
