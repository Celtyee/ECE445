from wunderground_crawler import *
import datetime
import os

import time


def daily_update_weather(driver_path, save_folder, context_len=30):
    cwl = weather_crawler(driver_path, save_folder)
    today = datetime.date.today() - datetime.timedelta(days=1)
    check_start_day = today - datetime.timedelta(days=context_len)

    # create the list of request date list
    date_list = [d for d in pd.date_range(check_start_day, today, freq='D')]
    if not os.path.exists(cwl.daily_weather_folder):
        os.mkdir(cwl.daily_weather_folder)
    for root, dirs, files in os.walk(cwl.daily_weather_folder):
        for date in date_list:
            i = datetime.datetime.strftime(date, '%Y-%m-%d')
            file_name = f'weather{i}.csv'
            # find the files, if not, get the csv file of this date.
            if file_name in files:
                continue
            date = datetime.datetime.strftime(date, '%Y%m%d')
            cwl.get_daily_weather(date, date)
            print(f'get the weather on day {i}')


def main():
    google_driver_path = "chromedriver_win32/chromedriver-112.exe"
    history_weather_path = "../../data/weather/history"
    if not os.path.exists(history_weather_path):
        os.mkdir(history_weather_path)
    daily_update_weather(google_driver_path, history_weather_path)


if __name__ == "__main__":
    # sleep the operating system for one day
    while True:
        main()
        time.sleep(86400)
