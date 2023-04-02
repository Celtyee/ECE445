from forecast import *

# XXX: num_day_pred should be less than 7 and larger than 0.
#   Don't change the value of pred_day.

model_path = './my_model/epoch=23-step=1200.ckpt'
pred_day = str(20221124)
num_day_context = 14
num_day_pred = 7
predict_api(model_path, pred_day, num_day_context, num_day_pred)
