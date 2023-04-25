import os

model_name_list = os.listdir("../data/train_recorder/")
for model_name in model_name_list:
    cmd = f"python test_module.py {model_name}"
    print(cmd)
    os.system(cmd)
