#!/usr/bin/env bash
# v14 full chain: check -> stage0 -> stage1 -> stage3 -> stage4 -> pricing (unattended)
set -e
cd "$(dirname "$0")/.."
CFG=config_v14.json
mkdir -p results/v14/logs
LOG=results/v14/logs/chain_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[chain] start $(date '+%F %T')"
python run_research.py --config $CFG check
python run_research.py --config $CFG stage0
python run_research.py --config $CFG stage1
python run_research.py --config $CFG stage3
python run_research.py --config $CFG stage4
python run_research.py --config $CFG pricing
echo "[chain] CHAIN_DONE $(date '+%F %T')"
