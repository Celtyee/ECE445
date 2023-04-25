import os
import sys

task_name = sys.argv[1]
model_name_list = os.listdir(f"../data/train/{task_name}/")
for model_name in model_name_list:
    cmd = f"python test_module.py {model_name} {task_name}"
    print(cmd)
    os.system(cmd)
