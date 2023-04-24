import os.path

from pred_api import prediction_api
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import matplotlib.pyplot as plt


def metrics_func(pred_day, data):
    '''

    Parameters
    ----------
    pred_day: The start day of prediction, str - "%Y%m%d".
    data: The json file saving the prediction result, dictionary.

    Returns
    -------
    rmse: the rmse for each building, list.
    mape: the mape for each buliding, list.
    mae: the mae for each buliding, list.

    '''
    rmse = []
    mape = []
    mae = []
    for building in data.keys():
        usage_pred = np.array(data[building])
        start_date = datetime.datetime.strptime(pred_day, "%Y%m%d")
        end_date = start_date + datetime.timedelta(days=7)
        building_path = f"../data/electricity/{building}.csv"
        df_electricity = pd.read_csv(building_path)
        df_electricity['time'] = pd.to_datetime(df_electricity['time']) + datetime.timedelta(hours=8)

        mask_ele = (df_electricity['time'].dt.date >= start_date.date()) & (
                df_electricity['time'].dt.date <= end_date.date())

        usage_y = df_electricity[mask_ele]['val'][1:]
        usage_y = np.array(usage_y)[:24 * 7]
        rmse.append(np.sqrt(mean_squared_error(usage_y, usage_pred)))
        mape.append(np.mean(np.abs(usage_y - usage_pred) / usage_y))
        mae.append(mean_absolute_error(usage_y, usage_pred))

    return rmse, mape, mae


def test(model_name):
    '''
    Test the performance of model on the test data set from 20210315 - 20210430
    Parameters
    ----------
    model_name: The name of model, str.
    Returns
    -------

    '''
    # select the ".ckpt" file path under the folder f"../data/train_recorder/{model_name}"
    model_path = os.path.join(f"../data/train_recorder/{model_name}",
                              [name for name in os.listdir(f"../data/train_recorder/{model_name}") if
                               name.endswith(".ckpt")][0])
    # data prediction for building 1A.
    buildings = ['1A']
    # 20210315 - 20210430
    pred_date_start = datetime.datetime.strptime("20210315", "%Y%m%d")

    # pred_date_end = datetime.datetime.strptime("20210317", "%Y%m%d")
    pred_date_end = datetime.datetime.strptime("20210430", "%Y%m%d")

    # create a datetime list from pred_date_start to pred_date_end
    pred_date_list = [pred_date_start + datetime.timedelta(days=i) for i in
                      range((pred_date_end - pred_date_start).days + 1)]
    csv_data = np.zeros((len(buildings), len(pred_date_list), 3))
    logging.basicConfig(filename=f"metrics/test.log",
                        level=logging.INFO)

    logger = logging.getLogger("Metrics:\n")
    for i in range(len(pred_date_list)):
        pred_date = pred_date_list[i]
        # print(pred_date)
        pred_day = pred_date.strftime("%Y%m%d")

        context_len = 30
        predction_len = 7
        num_day_context = context_len + predction_len

        weather_start_date = pred_date - datetime.timedelta(days=num_day_context + 1)
        weather_start_date = weather_start_date.strftime("%Y%m%d")

        prediction = prediction_api()
        prediction_result = prediction.custom_prediction(model_path, pred_day, weather_start_date)
        rmse_list, mape_list, mae_list = metrics_func(pred_day, prediction_result)
        # set the logger name as "metrics"
        for idx in range(len(buildings)):
            rmse = rmse_list[idx]
            mape = mape_list[idx]
            mae = mae_list[idx]
            csv_data[idx, i, 0] = rmse
            csv_data[idx, i, 1] = mape
            csv_data[idx, i, 2] = mae
            logger.info(f"{buildings[idx]}: RMSE: {rmse}, MAPE: {mape}, MAE: {mae}")

    # draw the graph of the RMSE, MAPE, MAE for 10 buildings
    metrics_list = ["RMSE", "MAPE", "MAE"]
    test_folder_path = f"../data/test/{model_name}"
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
    for m in range(3):
        plt.figure()
        for i in range(len(buildings)):
            plt.plot(pred_date_list, csv_data[i, :, m], label=buildings[i])
            plt.title(f"{metrics_list[m]}")
            plt.xlabel("Prediction Day")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
        plt.savefig(f"{test_folder_path}/{metrics_list[m]}.png")


if __name__ == "__main__":
    model_name_list = os.listdir("../data/train_recorder/")
    for model_name in model_name_list:
        test(model_name)
