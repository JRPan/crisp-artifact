# Artifacts for CRISP: Concurrent Rendering and Compute Simulation Platform for GPUs

### Software dependencies

The framework is tested on Ubuntu 20.04. The host computer should have Docker installed. CUDA is required to run the tracer.

- gcc/g++-9
- CUDA-11 (11.4 tested)
- Embree v3.12.0
- Vulkan SDK 1.2.162 or preferably newer
- Docker
- [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)

For Vulkan Samples, you can check out an earlier commit if you are experiencing CMake version issues. We used commit `2ce8856`.

### Data sets

All traces evaluated in the paper are provided. However, by default, only SPL paired with VIO is downloaded. If the user wishes to evaluate all workloads used in the paper, please follow the instructions accordingly.

### Installation

After all software dependencies are met, please run the following to install the remaining dependencies:

```bash
$ sudo apt install -y build-essential git ninja-build meson libboost-all-dev xutils-dev bison zlib1g-dev flex libglu1-mesa-dev libxi-dev libxmu-dev libdrm-dev llvm libelf-dev libwayland-dev wayland-protocols libwayland-egl-backend-dev libxcb-glx0-dev libxcb-shm0-dev libx11-xcb-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxshmfence-dev libxxf86vm-dev libxrandr-dev libglm-dev libelf-dev
```

Download and decompress the source codes: [https://zenodo.org/records/12803388](https://zenodo.org/records/12803388)

The source code contains 3 folders:

- **accel-sim-framework:** the simulator.
- **vulkan-sim:** the tracer.
- **mesa-vulkan-sim:** the Mesa 3D driver.

Next, set up environments:

```bash
$ export CUDA_INSTALL_PATH=/usr/local/cuda
$ cd crisp-framework
$ source vulkan-sim/setup_environment
```

Build the tracer. Please ignore the error in the first ninja build:

```bash
$ cd mesa-vulkan-sim
$ meson --prefix="${PWD}/lib" build/
$ meson configure build/ -Dbuildtype=debug -D b_lundef=false
$ ninja -C build/ install
$ cd ../vulkan-sim/
$ make -j
$ cd ../mesa-vulkan-sim
$ ninja -C build/ install
```

Build the Simulator within the Docker (from the crisp-framework folder):

```bash
$ docker run -it --rm -v $(pwd)/accel-sim-framework:/accel-sim/accel-sim-framework tgrogers/accel-sim_regress:Ubuntu-22.04-cuda-11.7
$ cd accel-sim-framework
$ source gpu_simulator/setup_environment
$ make -j -C ./gpu-simulator
$ exit
```

Finally, copy files `gpgpusim.config` and `config_turing_islip.icnt` from the crisp-framework folder to the Vulkan-Samples folder.

### Experiment workflow

To run the simulation:

```bash
$ docker run -it --rm -v $(pwd)/accel-sim-framework:/accel-sim/accel-sim-framework tgrogers/accel-sim_regress:Ubuntu-22.04-cuda-11.7
$ cd accel-sim-framework
$ . get_crisp_traces.sh
$ cd util/graphics
$ python3 ./setup_concurrent.py
$ cd ../../
$ . run.sh
```

The user can use `./util/job_launching/job_status.py` to monitor the simulation. The simulations are expected to run for 8 hours. After simulations are finished, the results are included in the folder `sim_run_11.7`.

To collect the stats, simply run the following command and then exit the Docker image:

```bash
$ . collect.sh
$ exit
```

Several CSV files should be generated under the accel-sim-framework folder. These files contain simulation statistics such as execution cycles and cache hit rates.

(optional) To collect traces, execute the command from the Vulkan-Sample folder:

```bash
$ VULKAN_APP=render_passes ./build/linux/app/bin/Release/x86_64/vulkan_samples sample render_passes
```

Then wait for the tracer to finish. The resolution has been changed to 480P to speed up the process. After the process is finished, a file called `complete.traceg` should be generated in the working directory.

Then, from the `accel-sim-framework/util/graphics` folder, edit line 7 to the `compelete.g` file, and optionally edit line 4 to change the output folder name.

```bash
$ python3 ./process-vulkan-traces.py
```

### Evaluation and expected results

The CSV files generated in the previous section contain all data used in this paper. The following scripts are used to generate figures in Section \ref{methodology}.

After completing the previous step, you should have the following CSV files under `accel-sim-framework`: `render_passes_2k.csv` and `render_passes_2k_lod0.csv`.

To generate the L1 texture plot similar to Figure L1 TEX Loads:

```bash
$ python3 ./util/graphics/l1tex.py
```

To generate the L2 breakdown plot similar to Figure L2 breakdown, change `./util/graphics/l2breakdown.py::7` to match the visualizer log file. The log files are generated under `sim_run*` folders.

```bash
$ python3 ./util/graphics/l2breakdown.py
```

To generate the concurrent ratio plot similar to Figure slicer occupancy, change `./util/graphics/concurrent_ratio.py::7` to match the visualizer log. For this one, please choose the one under `sim_run_11.7/render_passes_2k/all1/RTX3070-SASS-concurrent-fg-VISUAL`.

```bash
$ python3 ./util/graphics/concurrent_ratio.py
```

We provided a `.ipynb` notebook `util/graphics/working_set.ipynb` to perform static analysis as seen in Figure TEX working set.
