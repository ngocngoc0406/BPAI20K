#!/usr/bin/env bash
# v3.3 controlled stress suite (sequential so runs do not compete for the GPU). Results: experiments/live/, experiments/latest/
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DEMO=$($PY -c "import sys; sys.path.insert(0,'.'); from src.data.dataset_loader import index_videos; print(index_videos('data/raw/nexar')['00288'])")
echo "=== demo_30min_record $(date +%H:%M:%S)"
$PY scripts/live_benchmark.py --name demo_30min_record --source "file:$DEMO" --minutes 30 --profile sensitive --audio-volume 0 \
    --data-kind demo_video --full-video --keep-awake 2>&1 | grep -E "RESULT|Error|Traceback|line [0-9]+" | cut -c1-400
echo "=== light_change robustness $(date +%H:%M:%S)"
$PY scripts/run_robustness.py --model-dir models/safedrive_v3 --out robustness_v3_light_change.json --conditions clean,light_change --keep-awake 2>&1 | grep -vi warning | tail -5
echo "=== done $(date +%H:%M:%S)"
