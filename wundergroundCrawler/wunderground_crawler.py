'''
20211010更新
1. 使用 date_range方法创建查询时间序列query_list
2. dropna(axis=0, how='all')方法可以删除值全为空的行
'''

import requests
import datetime
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# clash port
PORT = 7890

# 输入查询的起止日期
# start_date = input('请输入查询天气起始日期（8位数字）：')
start_date = '20200731'
end_date = '20230308'
# end_date = input('请输入查询天气终止日期（8位数字）：')
# 处理起止日期
struct_start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
struct_end_date = datetime.datetime.strptime(end_date, '%Y%m%d')

proxies = {'http': 'http://127.0.0.1:{}'.format(PORT), 'https': 'http://127.0.0.1:{}'.format(PORT)}


def download_html_file(url, save_path="current_file.html"):
    # 实例化 webdriver 对象
    driver = webdriver.Chrome(executable_path="D:\chromedriver_win32\chromedriver.exe")
    driver.get(url)
    elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                           '''//*[@id="inner-content"]/div[2]/div[1]/div[5]/div[1]/div/lib-city-history-observation/div/div[2]''')))
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    driver.quit()


def read_weather_table(file_path="current_file.html"):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_string = f.read()

    # 使用pandas的read_html函数解析HTML表格
    table = pd.read_html(html_string)
    weather = table[0]
    weather = weather.dropna(axis=0, how='all')
    return weather


def get_weather(url):
    # 根据url查询历史天气
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36 Edg/93.0.961.52'
    }
    # 两种方法可以获得html
    # 方法1 requests库
    print("start crawling")
    resp = requests.get(url, headers=headers, proxies=proxies)
    if resp.status_code == 200:
        download_html_file(url)
        weather = read_weather_table()
        # 爬下来的数据中由于50°F是以英寸-磅-秒（imperial unit）为单位的温度，而不是以摄氏度或华氏度为单位，是包含特殊字符的
        return True, weather
    else:
        return False  # 如果异常，获取失败，返回None

    # 方法2 requests_html库
    # session = HTMLSession()
    # resp = session.get(url, headers=headers)
    # html = resp.html.html


def generate_url(date):
    # 创建查询url
    # 分析查询网页的url，每次需要改变两个位置，个人天气站点是ISHANG46
    # https://www.wunderground.com/history/daily/ZSHC/date/2023-3-2
    str_date = date.strftime('%Y-%m-%d')
    # date_obj = datetime.datetime.strptime(str_date, "%Y-%m-%d")
    # print(date_obj)
    # formatted_date = date_obj.strftime("%Y-%-m-%-d")
    weather_station = 'ZSHC'
    url = "https://www.wunderground.com/history/daily/{0}/{1}/{2}".format(weather_station, 'date', str_date)
    print()
    print(url)
    return url


def get_all_weather():
    '''
    # 获取查询日期区间的所有天气
    query_list = []
    query_days = (struct_end_date - struct_start_date).days + 1


    # 得到查询的日期序列query_list
    for i in range(query_days):
        query_list.append(struct_start_date + datetime.timedelta(days=i))
    '''

    query_list = [d for d in pd.date_range(struct_start_date, struct_end_date, freq='D')]
    df_list = []
    for j in tqdm(range(len(query_list))):
        i = query_list[j]
        date = i.strftime('%Y-%m-%d')
        res, df_get = get_weather(generate_url(i))
        if res:
            # df_get=df_get[~(df_get['Time'].isnull())] # 删除Time列为空NaN的行
            df_get.insert(0, 'Date', date)
            # df_get['Date'] = date
            df_list.append(df_get)
        else:
            print(date + '获取失败')
    df = pd.concat(df_list)
    start_date = query_list[0].strftime('%Y-%m-%d')
    end_date = query_list[-1].strftime('%Y-%m-%d')
    df.to_csv('weatherData/weather' + start_date + 'TO' + end_date + '.csv', index=False)
    return


get_all_weather()
