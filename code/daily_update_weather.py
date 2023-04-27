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


def main():
    # google_driver_path = "./wunderground_crawler/chromedriver_win32/chromedriver-112.exe"
    # history_weather_path = "../data/weather/history"
    # electricity_path = "../data/electricity"
    # if not os.path.exists(history_weather_path):
    #     os.mkdir(history_weather_path)
    # daily_update_weather(google_driver_path, history_weather_path)
    # generator = dataset_generator(history_weather_path, electricity_path)
    # generator.compress_weather_data(f'{history_weather_path}/pre-processed_weather.csv')
    vc = visualcrossing_crawler()
    start_date = datetime.datetime.strptime('2021-01-01', '%Y-%m-%d').date()
    # end date is today
    end_date = datetime.datetime.today().date()
    weather_path = "../data/weather/pre-processed_weather.csv"
    while True:
        vc.fetch_history(start_date, end_date, weather_path)
        # sleep the operating system for one day
        print("finished! sleep for one day.")
        time.sleep(86400)


if __name__ == "__main__":
    main()
