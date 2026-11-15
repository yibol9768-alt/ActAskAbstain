"""Write a task yaml of runs whose latest attempt has a connection/timeout error. usage: retry_errors.py <tasks_yaml> <results_model_dir> <out_yaml>"""
import json, sys, yaml
from pathlib import Path
tasks = yaml.safe_load(open(sys.argv[1]))["tasks"]; root = Path(sys.argv[2]); bad = []
for t in tasks:
    d = root / t["category"] / t["task_id"] / t["task_type"]
    runs = sorted(p for p in d.glob("*") if (p / "eval.json").exists()) if d.exists() else []
    err = json.load(open(runs[-1] / "eval.json")).get("run_error") if runs else "missing"
    if err and ("Connection" in err or "timed out" in err or err == "missing"):
        bad.append(t)
yaml.safe_dump({"tasks": bad}, open(sys.argv[3], "w"), sort_keys=False); print(len(bad), "to retry")
