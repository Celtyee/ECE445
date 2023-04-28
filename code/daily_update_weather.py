from wunderground_crawler.utils import visualcrossing_crawler
from utils import dataset_generator
import datetime
import os
import pandas as pd
import time


# def daily_update_weather(driver_path, save_folder, context_len=30):
#     cwl = wunderground_crawler(driver_path, save_folder)
#     today = datetime.date.today() - datetime.timedelta(days=1)
#     check_start_day = today - datetime.timedelta(days=context_len)
#
#     # create the list of request date list
#     date_list = [d for d in pd.date_range(check_start_day, today, freq='D')]
#     if not os.path.exists(cwl.daily_weather_folder):
#         os.mkdir(cwl.daily_weather_folder)
#     for root, dirs, files in os.walk(cwl.daily_weather_folder):
#         for date in date_list:
#             i = datetime.datetime.strftime(date, '%Y-%m-%d')
#             file_name = f'weather{i}.csv'
#             # find the files, if not, get the csv file of this date.
#             if file_name in files:
#                 continue
#             date = datetime.datetime.strftime(date, '%Y%m%d')
#             cwl.get_daily_weather(date, date)
#             print(f'get the weather on day {i}')


def fetch_history_weather():
    # google_driver_path = "./wunderground_crawler/chromedriver_win32/chromedriver-112.exe"
    # history_weather_path = "../data/weather/history"
    # electricity_path = "../data/electricity"
    # if not os.path.exists(history_weather_path):
    #     os.mkdir(history_weather_path)
    # daily_update_weather(google_driver_path, history_weather_path)
    # generator = dataset_generator(history_weather_path, electricity_path)
    # generator.compress_weather_data(f'{history_weather_path}/history_weather_wc.csv')

    vc = visualcrossing_crawler()
    # end date is today
    # create a list from 2021-01-01 to 2023-05-01 month by month
    start_date = datetime.datetime.strptime('2022-04-30', '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime('2023-04-01', '%Y-%m-%d').date()
    date_list = [d for d in pd.date_range(start_date, end_date, freq='M')]
    monthly_path = "../data/weather/monthly"
    if not os.path.exists(monthly_path):
        os.mkdir(monthly_path)
    for i in range(len(date_list) - 1):
        start_date = date_list[i].date()
        end_date = (date_list[i + 1] - datetime.timedelta(days=1)).date()
        save_path = f'{monthly_path}/{start_date}_{end_date}.csv'
        vc.fetch_history(start_date, end_date, save_path)


if __name__ == "__main__":
    # fetch_history_weather()
    # compress the data in monthly folder
    weather_monthly_path = "../data/weather/monthly"
    total_weather_df = pd.DataFrame()
    # for all files in weather_monthly_path
    for root, dirs, files in os.walk(weather_monthly_path):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            total_weather_df = pd.concat([total_weather_df, df], axis=0)
    history_weather_path = "../data/weather/history"
    total_weather_df.to_csv(f'{history_weather_path}/history_weather_vc.csv', index=False)
