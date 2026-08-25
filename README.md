# CRISP Artifact — Vulkan-Sim 2.0 Branch

> **This branch (`2.0`) tracks ongoing work to port CRISP onto Vulkan-Sim 2.0.**
> For the exact code used in the paper, see the [`main`](https://github.com/JRPan/crisp-artifact/tree/main) branch (aging, minimal support) or the pinned Zenodo release. If you need the original artifact and can't reproduce from `main`, please open an issue and I can help.

## Update — August 2026

- Migrated `vulkan-sim` and `mesa-vulkan-sim` to git submodules pointing at [`dev-2.0`](https://github.com/JRPan/vulkan-sim/tree/dev-2.0) and [`dev-2.0`](https://github.com/JRPan/mesa-vulkan-sim/tree/dev-2.0). `accel-sim-framework` is unchanged in this repo.
- Vulkan-Sim 2.0 uses **Lavapipe** as the driver — no Intel CPU required.
- [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples) is now the primary workload driver. Working samples so far: **sponza**, **instancing**, **PBR** (more being validated).
- The new NIR→PTX translator resolves several limitations of the paper version, so more workloads should be reachable going forward.
- Not everything is ported yet. If you hit issues, please file an issue.

---

# Artifacts for CRISP: Concurrent Rendering and Compute Simulation Platform for GPUs

### Repository layout

- `accel-sim-framework/` — the simulator (checked in, unchanged).
- `vulkan-sim/` — **git submodule**, tracks `dev-2.0` of [JRPan/vulkan-sim](https://github.com/JRPan/vulkan-sim/tree/dev-2.0).
- `mesa-vulkan-sim/` — **git submodule**, tracks `dev-2.0` of [JRPan/mesa-vulkan-sim](https://github.com/JRPan/mesa-vulkan-sim/tree/dev-2.0).
- `embree-3.13.5.x86_64.linux/` — Embree binary distribution.
- `gpgpusim.config`, `config_turing_islip.icnt` — simulator configs.

### Software dependencies

Tested on Ubuntu 20.04 / 22.04. Docker is required for the simulator build; CUDA is required for tracing.

- gcc/g++-9
- CUDA-11 (11.4 tested)
- Embree v3.13.5 (included)
- Vulkan SDK 1.2.162 or newer
- Docker
- [Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples)

### Cloning

Clone recursively so the submodules come with you:

```bash
$ git clone --recurse-submodules -b 2.0 https://github.com/JRPan/crisp-artifact.git
$ cd crisp-artifact
```

If you already cloned without `--recurse-submodules`:

```bash
$ git submodule update --init --recursive
```

To pull the latest changes on both submodule branches:

```bash
$ git submodule update --remote
```

### Installation

Install system packages:

```bash
$ sudo apt install -y build-essential git ninja-build meson libboost-all-dev xutils-dev bison zlib1g-dev flex libglu1-mesa-dev libxi-dev libxmu-dev libdrm-dev llvm libelf-dev libwayland-dev wayland-protocols libwayland-egl-backend-dev libxcb-glx0-dev libxcb-shm0-dev libx11-xcb-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxshmfence-dev libxxf86vm-dev libxrandr-dev libglm-dev libelf-dev
```

Set up the environment:

```bash
$ export CUDA_INSTALL_PATH=/usr/local/cuda
$ source vulkan-sim/setup_environment
```

Build order matters — **build `vulkan-sim` first, then Mesa** (Mesa links against symbols from vulkan-sim):

```bash
$ cd mesa-vulkan-sim
$ meson --prefix="${PWD}/lib" build/
$ meson configure build/ -Dbuildtype=debug -D b_lundef=false
$ ninja -C build/ install   # first pass may error; that's expected
$ cd ../vulkan-sim/
$ make -j$(nproc)
$ cd ../mesa-vulkan-sim
$ ninja -C build/ install
```

Register the Lavapipe ICD so applications load the vulkan-sim driver:

```bash
$ export VK_ICD_FILENAMES=$PWD/lib/share/vulkan/icd.d/lvp_icd.x86_64.json
```

Build the simulator inside Docker:

```bash
$ docker run -it --rm -v $(pwd)/accel-sim-framework:/accel-sim/accel-sim-framework tgrogers/accel-sim_regress:Ubuntu-22.04-cuda-11.7
$ cd accel-sim-framework
$ source gpu_simulator/setup_environment
$ make -j -C ./gpu-simulator
$ exit
```

Copy `gpgpusim.config` and `config_turing_islip.icnt` from this folder into the Vulkan-Samples build folder before running.

### Running Vulkan-Samples

Working samples on `dev-2.0` as of this update: `sponza`, `instancing`, `PBR`.

```bash
$ VULKAN_APP=instancing ./build/linux/app/bin/Release/x86_64/vulkan_samples sample instancing
```

The tracer writes `complete.traceg` in the working directory when the sample exits.

### Experiment workflow (paper reproduction)

The scripts below reproduce the paper experiments using the traces from the original artifact. Trace collection with 2.0 is still being validated across all paper workloads.

```bash
$ docker run -it --rm -v $(pwd)/accel-sim-framework:/accel-sim/accel-sim-framework tgrogers/accel-sim_regress:Ubuntu-22.04-cuda-11.7
$ cd accel-sim-framework
$ . get_crisp_traces.sh
$ cd util/graphics
$ python3 ./setup_concurrent.py
$ cd ../../
$ . run.sh
```

Monitor with `./util/job_launching/job_status.py`. Expect ~8 hours. Results land in `sim_run_11.7`.

Collect stats and exit the container:

```bash
$ . collect.sh
$ exit
```

### Collecting your own traces

From the Vulkan-Samples folder:

```bash
$ VULKAN_APP=render_passes ./build/linux/app/bin/Release/x86_64/vulkan_samples sample render_passes
```

Resolution is set to 480P for speed. Once `complete.traceg` is produced, from `accel-sim-framework/util/graphics/`, edit line 7 to point at the trace and (optionally) line 4 to set the output folder, then:

```bash
$ python3 ./process-vulkan-traces.py
```

### Evaluation

The scripts below reproduce paper figures from the collected CSVs (`render_passes_2k.csv`, `render_passes_2k_lod0.csv`).

L1 texture plot (Figure "L1 TEX Loads"):

```bash
$ python3 ./util/graphics/l1tex.py
```

L2 breakdown (Figure "L2 breakdown") — update `l2breakdown.py::7` to match the visualizer log under `sim_run*`:

```bash
$ python3 ./util/graphics/l2breakdown.py
```

Slicer occupancy (Figure "slicer occupancy") — update `concurrent_ratio.py::7` to `sim_run_11.7/render_passes_2k/all1/RTX3070-SASS-concurrent-fg-VISUAL`:

```bash
$ python3 ./util/graphics/concurrent_ratio.py
```

Static TEX working-set analysis (Figure "TEX working set") is in `util/graphics/working_set.ipynb`.

### Want the original paper artifact?

The exact snapshot used for the CRISP paper lives on the [`main`](https://github.com/JRPan/crisp-artifact/tree/main) branch and on [Zenodo](https://zenodo.org/records/12803388). Both are aging and largely unsupported — if you need help getting that version running, please open an issue and I'll do what I can.
