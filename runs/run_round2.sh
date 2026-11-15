#!/bin/bash
R=/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/runs
$R/run_gate.sh none gate40
$R/run_gate.sh gatev gate40
$R/run_gate.sh gate gateerr
