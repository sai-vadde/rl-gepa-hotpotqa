#!/bin/bash
# Full experiment on macOS/Linux, sequential and resumable. Logs go to logs/.
# usage: MODEL=qwen3:4b-instruct-2507 bash run_pipeline.sh
cd "$(dirname "$0")"
mkdir -p logs
PY=${PY:-.venv/bin/python}
M=${MODEL:-qwen3:4b-instruct-2507}
$PY -u run_all.py --stage pilot    --model "$M" >> logs/pilot.log 2>&1
$PY -u run_all.py --stage split                 >> logs/split.log 2>&1
$PY -u run_all.py --stage baseline --model "$M" >> logs/baseline.log 2>&1
$PY -u run_all.py --stage mipro    --model "$M" >> logs/mipro.log 2>&1
$PY -u run_all.py --stage gepa     --model "$M" --budget 900 --seed 11 >> logs/gepa_seed11.log 2>&1
$PY -u run_all.py --stage grpo     --model "$M" >> logs/grpo.log 2>&1   # needs a CUDA GPU
$PY -u run_all.py --stage tables                >> logs/tables.log 2>&1
echo ALL_DONE >> logs/pipeline.log
