#!/usr/bin/env bash

set -u
cd "$(dirname "$0")"

VENV=".venv/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "[pipeline] start $(date)"

"$VENV" -m scripts.build_tier2 > logs/tier2.log 2>&1 &
PID2=$!
"$VENV" -m scripts.ingest_datikz --target 1000 > logs/tier3.log 2>&1 &
PID3=$!

wait "$PID2"; RC2=$?
wait "$PID3"; RC3=$?
echo "[pipeline] tier2 exit=$RC2, tier3 exit=$RC3"

"$VENV" -m scripts.evaluate_multi_tier > logs/eval.log 2>&1
echo "[pipeline] eval exit=$?"

"$VENV" -m scripts.train_mixed > logs/train.log 2>&1
echo "[pipeline] train exit=$?"

echo "[pipeline] DONE $(date)"
