#!/bin/bash
source ./gpu-simulator/setup_environment.sh
./util/job_launching/run_simulations.py -B crisp-artifact:render_passes_2k_lod0 -C ORIN-SASS-concurrent-fg-VISUAL,RTX3070-SASS-concurrent-fg-VISUAL -T ./hw_run/traces/vulkan/ -N run-20240723-1728-render_passes_2k_lod0
./util/job_launching/run_simulations.py -B crisp-artifact:render_passes_2k -C ORIN-SASS-concurrent-fg-VISUAL,RTX3070-SASS-concurrent-fg-VISUAL -T ./hw_run/traces/vulkan/ -N run-20240723-1728-render_passes_2k
