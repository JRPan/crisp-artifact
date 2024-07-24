#!/bin/bash
name="-N run-20240723-1728"
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
./util/job_launching/get_stats.py -k -R $name-render_passes_2k_lod0 > render_passes_2k_lod0.csv &
./util/job_launching/get_stats.py -k -R $name-render_passes_2k > render_passes_2k.csv &
wait < <(jobs -p)
