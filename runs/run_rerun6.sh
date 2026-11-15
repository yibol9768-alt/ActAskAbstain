#!/bin/bash
# Operational pairs not in pilot24 (rerun 6 connection-error runs), DeepSeek, 2 concurrent.
set -a; . /mnt/c/Users/liuyibo/Desktop/auto-llm-driving/deepseek-key.env; set +a
cd /mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/external/agentabstain
PY=/home/liuyibo/venvs/actask/bin/python
R=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/runs
echo "start $(date)"
$PY $R/infer_ds.py --runtime-config src/configs/openaisdk_ds-v4.1-flash.yaml --task-config src/configs/tasks_rerun6.yaml --workers 2
echo "infer done $(date)"
$PY $R/eval_ds.py --provider openaisdk --model deepseek-v4.1-flash --judge-config eval/configs/ds_judge.yaml --workers 2
echo "eval done $(date)"
