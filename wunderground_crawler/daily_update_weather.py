from wunderground_crawler import *
import datetime
import os

driver_path = "D:/chromedriver_win32/chromedriver.exe"
cache_path = "./cache"

daily_weather_folder = "../../data/weather/history/daily_weather"
context_len = 30
cwl = weather_crawler(driver_path, cache_path)
today = datetime.date.today()
check_start_day = today - datetime.timedelta(days=context_len)

# create the list of request dates
# find the files, if not, get the csv file of this date.