from forecast import *

model_path = './my_model/epoch=23-step=1200.ckpt'
pred_day = str(20221124)
num_day_context = 30
num_day_pred = 7
predict_api(model_path, pred_day, num_day_context, num_day_pred)
