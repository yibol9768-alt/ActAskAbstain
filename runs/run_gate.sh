#!/bin/bash
# usage: run_gate.sh <mode gate|reflect> <taskset name>   (DeepSeek, 2 concurrent)
set -a; . /mnt/c/Users/liuyibo/Desktop/auto-llm-driving/deepseek-key.env; set +a
cd /mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/external/agentabstain
PY=/home/liuyibo/venvs/actask/bin/python
R=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/runs
M=$1; T=$2
echo "start $M $T $(date)"
ACT_MODE=$M ACT_TAG=${M}_$T $PY $R/infer_gate.py --runtime-config src/configs/openaisdk_ds-v4.1-flash_$M.yaml --task-config src/configs/tasks_$T.yaml --workers 2
echo "infer done $(date)"
$PY $R/eval_ds.py --provider openaisdk --model deepseek-v4.1-flash --judge-config eval/configs/ds_judge.yaml --results-root results_$M --workers 2
echo "eval done $(date)"
