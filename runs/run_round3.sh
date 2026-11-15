#!/bin/bash
# scale gatev + base2(none) to the 85 remaining operational pairs, then retry connection errors once (all of gate40+rest85)
R=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/runs
A=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/external/agentabstain
PY=/home/liuyibo/venvs/actask/bin/python
$R/run_gate.sh gatev rest85
$R/run_gate.sh none rest85
for M in gatev none; do
  for T in gate40 rest85; do
    $PY $R/retry_errors.py $R/tasks_$T.yaml $A/results_$M/openaisdk/deepseek-v4.1-flash $A/src/configs/tasks_retry_${M}_$T.yaml
    n=$(grep -c "task_id" $A/src/configs/tasks_retry_${M}_$T.yaml || true)
    [ "$n" -gt 0 ] && $R/run_gate.sh $M retry_${M}_$T
  done
done
echo "round3 all done $(date)"
