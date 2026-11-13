#!/usr/bin/env bash
# Full chain for any config: check -> stage0 -> stage1 -> stage3 -> stage4 -> pricing (unattended)
# usage: bash scripts/run_chain.sh config_v15.json
set -e
cd "$(dirname "$0")/.."
CFG=${1:-config_v15.json}
SUB=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['study']['results_subdir'])" "$CFG")
mkdir -p results/$SUB/logs
LOG=results/$SUB/logs/chain_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "[chain] start $(date '+%F %T') config=$CFG results=results/$SUB"
python run_research.py --config $CFG check
python run_research.py --config $CFG stage0
python run_research.py --config $CFG stage1
python run_research.py --config $CFG stage3
python run_research.py --config $CFG stage4
python run_research.py --config $CFG pricing
echo "[chain] CHAIN_DONE $(date '+%F %T')"
