import sys, runpy
sys.path.insert(0, "/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/runs")
sys.path.insert(0, "/mnt/c/Users/liuyibo/Desktop/auto-llm-driving/ActAskAbstain/external/agentabstain")
import ds_patch
ds_patch.patch_agents()
sys.argv = ["run_inference"] + sys.argv[1:]
runpy.run_module("src.scripts.run_inference", run_name="__main__")
