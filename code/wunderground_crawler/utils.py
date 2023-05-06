import datetime
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
import urllib.request
import sys
import csv
import codecs


class wunderground_crawler:
    def __init__(self, selenium_driver_path, weather_save_folder, port=7890):
        cache_path = "./crawler_cache"
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.html_path = f"{cache_path}/current_file.html"
        self.driver_path = selenium_driver_path
        self.weather_folder = f'{weather_save_folder}'
        self.daily_weather_folder = f'{self.weather_folder}/daily_weather'
        self.port = port

    def __download_html_file(self, url):
        # 实例化 webdriver 对象
        # hide the browser window
        chrome_opt = Options()
        chrome_opt.add_argument("--headless")
        driver = webdriver.Chrome(executable_path=self.driver_path, options=chrome_opt)
        driver.get(url)
        elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                               '''//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]''')))
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        driver.quit()

    def __read_weather_table(self):
        with open(self.html_path, 'r', encoding='utf-8') as f:
            html_string = f.read()

        # 使用pandas的read_html函数解析HTML表格
        table = pd.read_html(html_string)
        weather = table[0]
        weather = weather.dropna(axis=0, how='all')
        return weather

    def __generate_url(self, date):
        # 创建查询url
        # 分析查询网页的url，每次需要改变两个位置，个人天气站点是ISHANG46
        # https://www.wunderground.com/history/daily/ZSHC/date/2023-3-2
        str_date = date.strftime('%Y-%m-%d')
        # date_obj = datetime.datetime.strptime(str_date, "%Y-%m-%d")
        # print(date_obj)
        # formatted_date = date_obj.strftime("%Y-%-m-%-d")
        weather_station = 'ZSHC'
        url = "https://www.wunderground.com/history/daily/{0}/{1}/{2}".format(weather_station, 'date', str_date)
        # print()
        return url

    def get_oneday_weather(self, url):
        # 根据url查询历史天气
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36 Edg/93.0.961.52'
        }
        proxies = {'http': 'http://127.0.0.1:{}'.format(self.port), 'https': 'http://127.0.0.1:{}'.format(self.port)}
        resp = requests.get(url, headers=headers, proxies=proxies)
        if resp.status_code == 200:
            self.__download_html_file(url)
            weather = self.__read_weather_table()
            # 爬下来的数据中由于50°F是以英寸-磅-秒（imperial unit）为单位的温度，而不是以摄氏度或华氏度为单位，是包含特殊字符的
            return True, weather
        else:
            return False  # 如果异常，获取失败，返回None

    # start_date = '20210726'
    # end_date = '20230308'
    def get_daily_weather(self, start_date, end_date):
        '''
        # 获取查询日期区间的所有天气
        query_list = []
        query_days = (struct_end_date - struct_start_date).days + 1


        # 得到查询的日期序列query_list
        for i in range(query_days):
            query_list.append(struct_start_date + datetime.timedelta(days=i))
        '''
        # 处理起止日期
        if not os.path.exists(self.daily_weather_folder):
            os.makedirs(self.daily_weather_folder)
        struct_start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        struct_end_date = datetime.datetime.strptime(end_date, '%Y%m%d')

        query_list = [d for d in pd.date_range(struct_start_date, struct_end_date, freq='D')]
        df_list = []
        for j in range(len(query_list)):
            i = query_list[j]
            date = i.strftime('%Y-%m-%d')
            url = self.__generate_url(i)
            res, df_get = self.get_oneday_weather(url)
            if res:
                # df_get=df_get[~(df_get['Time'].isnull())] # 删除Time列为空NaN的行
                df_get.insert(0, 'Date', date)
                # df_get['Date'] = date
                df_list.append(df_get)
                df_backup = pd.concat([df_get])
                df_backup.to_csv(f'{self.daily_weather_folder}/weather{date}.csv', index=False)
            else:
                print(date + 'crawl fails')


class visualcrossing_crawler:
    def crawl_forecast(self, start_date, end_date, csv_save_path):
        '''
        crawl the forecast data from visualcrossing
        Parameters
        ----------
        csv_save_path:
        start_date: start_date of forecasting, datetime.datetime.date.
        end_date: end_date of forecasting, datetime.datetime.date.

        Returns
        -------
        The forecast dataframe, pandas dataframe.
        '''
        # turn into the form of 'yyyy-mm-dd'
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        try:
            ResultBytes = urllib.request.urlopen(
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/30.35%2C%20120.32/{start_date}/{end_date}?unitGroup=us&include=hours&key=WPBAQQSNZTMMARJ2TFEKBGYFL&contentType=csv")

            # Parse the results as CSV
            CSVText = csv.reader(codecs.iterdecode(ResultBytes, 'utf-8'))

        except urllib.error.HTTPError as e:
            ErrorInfo = e.read().decode()
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()
        except  urllib.error.URLError as e:
            ErrorInfo = e.read().decode()
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()
        # turn it into pandas dataframe
        df = pd.DataFrame(CSVText)
        # set column as the first row
        df.columns = df.iloc[0]
        df = df.drop(0)
        # turn the datetime into timestamp type
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={'datetime': 'timestamp',
                                'temp': 'Temperature',
                                'Dew Point': 'Dew Point',
                                'humidity': 'Humidity',
                                'conditions': 'Condition'})
        df.to_csv(csv_save_path, index=False)
        return csv_save_path

    def fetch_history(self, start_date, end_date, csv_save_path):
        '''

        Parameters
        ----------
        start_date: start date of crawling, datetime.datetime.time.
        end_date: end date of crawling, datetime.datetime.time.
        csv_save_path: path to save the csv file, str.

        Returns
        -------
        The history dataframe, pandas dataframe.
        '''
        # turn into the form of 'yyyy-mm-dd'
        print("fetch from {} to {}".format(start_date, end_date))
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')

        try:
            # liyang.19@intl.zju.edu.cn
            # ResultBytes = urllib.request.urlopen(
            #     f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/30.35%2C%20120.32/{start_date}/{end_date}?unitGroup=us&include=hours&key=HP5E7ZNHCRY47ANDBDJ7WEXDE&contentType=csv")

            # Aoz.19@intl.zju.edu.cn
            # ResultBytes = urllib.request.urlopen(
            #     f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/30.35%2C%20120.32/{start_date}/{end_date}?unitGroup=us&include=hours&key=SZXNC5GBHQA5GZD9DAAD9NFE5&contentType=csv")

            # chanleon1124@gmail.com
            ResultBytes = urllib.request.urlopen(
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/30.35%2C%20120.32/{start_date}/{end_date}?unitGroup=us&include=hours&key=WPBAQQSNZTMMARJ2TFEKBGYFL&contentType=csv")

            # Parse the results as CSV
            CSVText = csv.reader(codecs.iterdecode(ResultBytes, 'utf-8'))

        except urllib.error.HTTPError as e:
            ErrorInfo = e.read().decode()
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()
        except  urllib.error.URLError as e:
            ErrorInfo = e.read().decode()
            print('Error code: ', e.code, ErrorInfo)
            sys.exit()

        # turn it into pandas dataframe
        df = pd.DataFrame(CSVText)
        # set column as the first row
        df.columns = df.iloc[0]
        df = df.drop(0)
        # turn the datetime into timestamp type
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={'datetime': 'timestamp',
                                'temp': 'Temperature',
                                'Dew Point': 'Dew Point',
                                'humidity': 'Humidity',
                                'conditions': 'Condition'})

        df.to_csv(csv_save_path, index=False)
        return csv_save_path


def unit_test_vc():
    # test the visualcrossing_crawler
    vc = visualcrossing_crawler()
    start_date = datetime.datetime.strptime('20210724', '%Y%m%d')
    end_date = datetime.datetime.strptime('20210726', '%Y%m%d')
    vc.fetch_history(start_date, end_date, 'test_history.csv')

    # enda date is today
    end_date = datetime.datetime.today() - datetime.timedelta(days=1)
    vc.crawl_forecast(start_date, end_date, 'test_forecast.csv')


def unit_test_wc():
    # test the wunderground_crawler
    pass


if __name__ == "__main__":
    unit_test_vc()
