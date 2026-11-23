#!/usr/bin/env bash
# v3.2 live evaluation suite (sequential: one camera user at a time). Results: experiments/live/*.json
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DEMO=$($PY -c "import sys; sys.path.insert(0,'.'); from src.data.dataset_loader import index_videos; print(index_videos('data/raw/nexar')['00288'])")
run() { echo "=== $1 $(date +%H:%M:%S)"; $PY scripts/live_benchmark.py "$@" 2>&1 | grep -E "RESULT|Error|Traceback|line [0-9]+"; }
run --name cam_5min --source camera:0 --minutes 5 --no-record
run --name cam_15min --source camera:0 --minutes 15 --no-record
SAFEDRIVE_DEVICE=cpu run --name cam_cpu_only --source camera:0 --minutes 2 --no-record --device cpu
run --name cam_disconnect --source camera:0 --minutes 1.5 --no-record --simulate-disconnect 30
echo "=== cam_occupied $(date +%H:%M:%S)"
$PY scripts/live_occupied_test.py 2>&1 | grep -E "OCCUPIED|Error|Traceback"
run --name demo_record --source "file:$DEMO" --minutes 5 --profile sensitive --audio-volume 0
run --name demo_no_audio --source "file:$DEMO" --minutes 1 --profile sensitive --audio-backend none
run --name demo_disk_full --source "file:$DEMO" --minutes 1 --profile sensitive --min-free-mb 100000000
echo "=== done $(date +%H:%M:%S)"
