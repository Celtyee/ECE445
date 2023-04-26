import os
import sys

task_name = sys.argv[1]
prediction_len = sys.argv[2]
test_path = "../data/test"
if not os.path.exists(test_path):
    os.mkdir(test_path)
model_name_list = os.listdir(f"../data/train/{task_name}/")
for model_name in model_name_list:
    cmd = f"python test_module.py {model_name} {task_name} {prediction_len}"
    print(cmd)
    os.system(cmd)
