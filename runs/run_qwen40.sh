#!/bin/bash
R=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/runs
$R/run_gate_qwen.sh none gate40
$R/run_gate_qwen.sh gatev gate40
echo "qwen all done $(date)"
