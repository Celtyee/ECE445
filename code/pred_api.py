import datetime
import pandas as pd
import os
import json
from utils import dataset_generator, my_deepAR_model


class prediction_api:
    def lastest_prediction(self, model_path):
        '''
        forecast the electricity load from [pred_day, pred_day+num_day_pred). Save the prediction in ./prediction.json.
        The function will crawl data from google forecast website.

        Parameters
        ----------3
        model_path: pytorch checkpoint for training

        Returns
        -------
        '''

        num_day_context = 30
        num_day_pred = 7
        buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']

        pred_date_start = datetime.datetime.now().date()
        pred_date_end = pred_date_start + datetime.timedelta(days=num_day_pred - 1)

        hist_date_start = pred_date_start - datetime.timedelta(days=num_day_context + num_day_pred + 1)
        hist_date_end = pred_date_start - datetime.timedelta(days=1)

        print(f'forecast date {pred_date_start} - {pred_date_end}')
        print(f'historical date {hist_date_start} - {hist_date_end}')

        electricity_path = "../data/electricity"
        future_weather_path = "../data/weather/future"
        # TODO: use google forecast API to get the weather forecast data
        # if crawl_forecast:
        #     # crawl weather forecast data from wunderground
        #
        #     driver_path = "wunderground_crawler/chromedriver_win32/chromedriver-112.exe"
        #
        #     crawler = wunderground_crawler.weather_crawler(driver_path, future_weather_path)
        #     crawler.get_daily_weather(start_date=pred_date_start.strftime("%Y%m%d"),
        #                               end_date=pred_date_end.strftime("%Y%m%d"))

        # compress the future data. The data will be saved in future_weather_path/future_weather.csv
        pred_weather_csv = f'{future_weather_path}/future_weather.csv'
        future_generator = dataset_generator(future_weather_path, electricity_path)
        future_generator.compress_weather_data(pred_weather_csv)

        # get time_varying_known_real data for TimeSeriesDataSet

        pred_df_list = future_generator.generate_dataset(buildings, pred_date_start, pred_date_end, pred_weather_csv)

        # get historical whether data and electricity data
        history_weather_path = "../data/weather/history"

        hist_weather_csv = f'{history_weather_path}/pre-processed_weather.csv'
        historical_generator = dataset_generator(history_weather_path, electricity_path)

        hist_df_list = historical_generator.generate_dataset(buildings, hist_date_start, hist_date_end,
                                                             hist_weather_csv)

        # combine historical data and forecast weather data together as the condition data
        total_df_list = []
        for i in range(len(buildings)):
            pred_df_list[i]['val'] = 0
            df = pd.concat((hist_df_list[i], pred_df_list[i]), axis=0)
            df['time_idx'] = range(len(df))
            total_df_list.append(df)
        pred_data = pd.concat(total_df_list)
        save_folder_path = "../data/test"
        pred_data_path = f'{save_folder_path}/predict_data.csv'
        pred_data.to_csv(pred_data_path, index=False)

        # run prediction
        # print(f'read csv file from {pred_data_path}')
        model = my_deepAR_model(model_path, 24 * num_day_context, 24 * num_day_pred, buildings)
        prediction = model.predict(pred_data_path)

        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        prediction_path = "./prediction.json"
        with open(prediction_path, "w") as f:
            json.dump(prediction, f)

        print("Prediction finishes")
        return prediction

    def custom_prediction(self, model_path, pred_date, weather_date):
        '''
        allow users to use custom weather as the input data for the prediction of the start day.
        The prediction result is stored in the file history_prediction.json.

        Parameters
        -------
        model_path: the path of pytorch checkpoint file, str: %Y%m%d.
        pred_date: the start date of prediciton,  str. The pred_date should be one week ago from today.
        weather_date: the start date of custom input weather, str: %Y%m%d.

        Returns
        -------
        prediction: the prediction result of custom context weather condition data, dict.
        '''
        prediction_len = 7
        context_len = 30
        # generate the input dataset
        buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']

        pred_date_start = datetime.datetime.strptime(pred_date, "%Y%m%d").date()
        pred_date_end = pred_date_start + datetime.timedelta(days=prediction_len - 1)

        hist_date_start = datetime.datetime.strptime(weather_date, "%Y%m%d").date()
        hist_date_end = hist_date_start + datetime.timedelta(days=context_len + prediction_len)

        print(f'forecast date {pred_date_start} - {pred_date_end}')
        print(f'historical date {hist_date_start} - {hist_date_end}')

        history_weather_path = "../data/weather/history"
        electricity_path = "../data/electricity"
        hist_weather_csv = f'{history_weather_path}/pre-processed_weather.csv'
        historical_generator = dataset_generator(history_weather_path, electricity_path)

        hist_df_list = historical_generator.generate_dataset(buildings, hist_date_start, hist_date_end,
                                                             hist_weather_csv)

        pred_df_list = historical_generator.generate_dataset(buildings, pred_date_start, pred_date_end,
                                                             hist_weather_csv)

        # combine historical data and forecast weather data together as the condition data
        total_df_list = []
        for i in range(len(buildings)):
            pred_df_list[i]['val'] = 0
            df = pd.concat((hist_df_list[i], pred_df_list[i]), axis=0)
            df['time_idx'] = range(len(df))
            total_df_list.append(df)
        input_df = pd.concat(total_df_list)
        save_folder_path = "../data/test"
        pred_data_path = f'{save_folder_path}/predict_data.csv'
        input_df.to_csv(pred_data_path, index=False)
        # run prediction
        model = my_deepAR_model(model_path, 24 * context_len, 24 * prediction_len, buildings)
        prediction = model.predict(pred_data_path)

        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        prediction_path = "./historical_prediction.json"
        with open(prediction_path, "w") as f:
            json.dump(prediction, f)

        print("Prediction finishes")
        return prediction
