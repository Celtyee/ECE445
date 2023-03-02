#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
20211010更新
1. 使用 date_range方法创建查询时间序列query_list
2. dropna(axis=0, how='all')方法可以删除值全为空的行
'''

# In[2]:


import requests
import datetime
import pandas as pd
from tqdm import tqdm

# from requests_html import HTMLSession


# In[3]:


# 输入查询的起止日期
start_date = input('请输入查询天气起始日期（8位数字）：')
end_date = input('请输入查询天气终止日期（8位数字）：')
# 处理起止日期
struct_start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
struct_end_date = datetime.datetime.strptime(end_date, '%Y%m%d')


# In[4]:


def get_weather(url):
    # 根据url查询历史天气
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36 Edg/93.0.961.52'
    }
    # 两种方法可以获得html
    # 方法1 requests库
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        html = resp.text
        '''
        # 方法2 requests_html库
        session = HTMLSession()
        resp = session.get(url, headers=headers)
        html = resp.html.html
        '''
        df_list = pd.read_html(html)
        weather = df_list[3]  # 一个一个试出来，第四个表格为我需要的天气表格
        weather.dropna(axis=0, how='all')  # 删除值全为空的行
        return True, weather
    else:
        return False  # 如果异常，获取失败，返回None


def generate_url(date):
    # 创建查询url
    # 分析查询网页的url，每次需要改变两个位置，个人天气站点是ISHANG46
    # url = 'https://www.wunderground.com/dashboard/pws/ISHANG46/table/2021-10-9/2021-10-9/daily'
    str_date = date.strftime('%Y-%m-%d')
    weather_station = 'ISHANG46'
    url = 'https://www.wunderground.com/dashboard/pws/' + weather_station + '/table/' + str_date + '/' + str_date + '/daily'
    return url


# In[5]:


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
    print(query_list)
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
    df.to_csv('weather' + start_date + 'TO' + end_date + '.csv', index=0)
    return


# In[6]:


get_all_weather()

# In[7]:


df = pd.read_csv('weather2021-05-01TO2021-10-08.csv')
df
