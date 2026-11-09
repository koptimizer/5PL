#!/usr/bin/env bash
# v13 Baseline Stage 2B: hR sensitivity over full Delta (420 LPs), independent of the v14 chain
set -e
cd "$(dirname "$0")/.."
mkdir -p results/v13/logs
LOG=results/v13/logs/stage2b_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[stage2b] start $(date '+%F %T')"
python run_research.py --config config_baseline.json stage2b
echo "[stage2b] STAGE2B_DONE $(date '+%F %T')"
