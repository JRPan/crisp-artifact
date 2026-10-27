#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
grep -v '^#' "$1" | xargs apt-get install -y --no-install-recommends
apt-get clean && rm -rf /var/lib/apt/lists/*
