import datetime
import pandas as pd
import os
import json
from utils import dataset_generator, my_deepAR_model
from wunderground_crawler import wunderground_crawler


def predict_api(model_path, pred_day, num_day_context=30, num_day_pred=7, crawl_forecast=False):
    '''
    forecast the electricity load from [pred_day, pred_day+num_day_pred). Save the prediction in ../data/prediction/prediction.json

    Parameters
    ----------
    model_path: pytorch checkpoint for training
    pred_day: the beginning day of prediction, in the form "%Y%m%d", e.g., "20230308".
    num_day_context:
    num_day_pred: length of prediction, setting default as 7
    crawl_forecast: To determine whether to get the forecast data.

    Returns
    -------
    '''

    # max num for prediction is one week
    assert 7 >= num_day_pred > 0
    building = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']

    pred_date_start = datetime.datetime.strptime(pred_day, "%Y%m%d").date()
    pred_date_end = pred_date_start + datetime.timedelta(days=num_day_pred - 1)

    hist_date_start = pred_date_start - datetime.timedelta(days=num_day_context + num_day_pred + 1)
    hist_date_end = pred_date_start - datetime.timedelta(days=1)

    print(f'forecast date {pred_date_start} - {pred_date_end}')
    print(f'historical date {hist_date_start} - {hist_date_end}')

    electricity_path = "../data/electricity"
    future_weather_path = "../data/weather/future"
    if crawl_forecast:
        # crawl weather forecast data from wunderground

        driver_path = "wunderground_crawler/chromedriver_win32/chromedriver-112.exe"

        crawler = wunderground_crawler.weather_crawler(driver_path, future_weather_path)
        crawler.get_daily_weather(start_date=pred_date_start.strftime("%Y%m%d"),
                                  end_date=pred_date_end.strftime("%Y%m%d"))

    # compress the future data
    pred_weather_csv = f'{future_weather_path}/future_weather.csv'
    future_generator = dataset_generator(future_weather_path, electricity_path)
    future_generator.compress_weather_data(pred_weather_csv)

    # get time_varying_known_real data for TimeSeriesDataSet

    pred_df_list = future_generator.generate_dataset(building, pred_date_start, pred_date_end, pred_weather_csv)

    # get historical whether data and electricity data
    history_weather_path = "../data/weather/history"

    hist_weather_csv = f'{history_weather_path}/pre-processed_weather.csv'
    historical_generator = dataset_generator(history_weather_path, electricity_path)

    hist_df_list = historical_generator.generate_dataset(building, hist_date_start, hist_date_end, hist_weather_csv)

    # combine historical data and forecast weather data together as the condition data
    total_df_list = []
    for i in range(len(building)):
        pred_df_list[i]['val'] = 0
        df = pd.concat((hist_df_list[i], pred_df_list[i]), axis=0)
        df['time_idx'] = range(len(df))
        total_df_list.append(df)
    pred_data = pd.concat(total_df_list)
    pred_data_path = f'{future_weather_path}/predict_data.csv'
    pred_data.to_csv(pred_data_path, index=False)

    # run prediction
    print(f'read csv file from {pred_data_path}')
    model = my_deepAR_model(model_path, 24 * num_day_context, 24 * num_day_pred, building)
    prediction = model.predict(pred_data_path)

    save_folder_path = "../data/test"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    with open(f"{save_folder_path}/prediction.json", "w") as f:
        json.dump(prediction, f)

    print("Prediction finishes")
    return True
