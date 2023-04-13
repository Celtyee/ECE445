from wunderground_crawler import *
import datetime
import os


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
    google_driver_path = "./chromedriver_win32/chromedriver.exe"
    weather_folder = "../../data/weather/history"
    if not os.path.exists(weather_folder):
        os.mkdir(weather_folder)
    daily_update_weather(google_driver_path, weather_folder)


if __name__ == "__main__":
    main()
