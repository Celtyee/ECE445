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


def fetch_history_weather_monthly(start_date, end_date):
    '''

    Parameters
    ----------
    start_date: datetime.datetime.date(). The start date of fetch monthly weather
    end_date: datetime.date.date(). The end date of fetch monthly weather

    Returns
    -------

    '''
    # wunderground website
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
    date_list = [d for d in pd.date_range(start_date, end_date, freq='M')]
    monthly_path = "../data/weather/monthly"
    if not os.path.exists(monthly_path):
        os.mkdir(monthly_path)
    for i in range(len(date_list) - 1):
        start_date = date_list[i].date()
        end_date = (date_list[i + 1] - datetime.timedelta(days=1)).date()
        save_path = f'{monthly_path}/{start_date}_{end_date}.csv'
        vc.fetch_history(start_date, end_date, save_path)

    # save_path = f'{monthly_path}/{start_date}_{end_date}.csv'
    # vc.fetch_history(start_date, end_date, save_path)
    # compress the data in monthly folder

    total_weather_df = pd.DataFrame()
    # for all files in weather_monthly_path
    for root, dirs, files in os.walk(monthly_path):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            total_weather_df = pd.concat([total_weather_df, df], axis=0)
    history_weather_path = "../data/weather/history"
    total_weather_df.to_csv(f'{history_weather_path}/history_weather_vc.csv', index=False)


def fetch_weather_daily(fetch_date):
    '''
    Get the weather condition for a specific day. Merge the daily weather into the total weather.
    Parameters
    ----------
    fetch_date: datetime.datetime.date(). The date of specific day.

    Returns
    -------

    '''
    vc = visualcrossing_crawler()
    daily_path = "../data/weather/daily"
    if not os.path.exists(daily_path):
        os.mkdir(daily_path)
        print('create the daily folder')
    save_path = f'{daily_path}/weather{fetch_date}.csv'
    if not os.path.exists(save_path):
        df_fetch = vc.fetch_history(fetch_date, fetch_date, save_path)
    else:
        df_fetch = pd.read_csv(save_path)
    #     attach the daily weather to the total weather
    df_history_weather = pd.read_csv("../data/weather/history/history_weather_vc.csv")
    # set the date type as datetime.datetime
    df_history_weather['timestamp'] = pd.to_datetime(df_history_weather['timestamp'])
    df_history_old = df_history_weather[df_history_weather['timestamp'].dt.date < fetch_date]
    df_history_new = pd.concat([df_history_old, df_fetch], axis=0)
    df_history_new.to_csv('../data/weather/history/history_weather_vc.csv', index=False)


def unit_test_vc():
    # test the visualcrossing_crawler
    today = datetime.datetime.today()
    fetch_weather_daily(today.date())
    # enda date is today
    vc = visualcrossing_crawler()
    end_date = datetime.datetime.today() + datetime.timedelta(days=6)
    save_path = f"../data/weather/future/forecast_weather_{today.date()}_{end_date.date()}.csv"
    vc.crawl_forecast(today, end_date, save_path)


if __name__ == "__main__":
    unit_test_vc()
    pass
