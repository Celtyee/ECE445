from pred_api import prediction_api
import json
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


def main():
    model_path = "my_model/hidden=28-rnn_layer=2-context_day=30-min_lr=0.0001.ckpt"
    buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']
    # 20210315 - 20210430
    pred_date_start = datetime.datetime.strptime("20210315", "%Y%m%d")
    pred_date_end = datetime.datetime.strptime("20210430", "%Y%m%d")
    # create a datetime list from pred_date_start to pred_date_end
    pred_date_list = [pred_date_start + datetime.timedelta(days=i) for i in
                      range((pred_date_end - pred_date_start).days + 1)]
    csv_data = np.zeros((10, len(pred_date_list), 3))
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
        idx = 0
        for rmse, mape, mae in zip(rmse_list, mape_list, mae_list):
            csv_data[idx, i, 0] = rmse
            csv_data[idx, i, 1] = mape
            csv_data[idx, i, 2] = mae
            logger.info(f"{buildings[idx]}: RMSE: {rmse}, MAPE: {mape}, MAE: {mae}")
            idx += 1

    # draw the graph of the RMSE, MAPE, MAE for 10 buildings
    metrics_list = ["RMSE", "MAPE", "MAE"]
    for m in range(3):
        plt.figure()
        for i in range(10):
            plt.plot(pred_date_list, csv_data[i, :, m], label=buildings[i])
            plt.title(f"{metrics_list[m]}")
            plt.xlabel("Prediction Day")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
        plt.savefig(f"metrics/{metrics_list[m]}.png")


if __name__ == "__main__":
    main()
