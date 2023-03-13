import pandas as pd
import re


def read_data_from_csv():
    pass


def remove_unit(x):
    pattern = r"([-+]?\d*\.\d+|\d+)"  # 匹配数值部分的正则表达式
    match = re.search(pattern, x)  # 搜索匹配结果
    if match is not None:
        return float(match.group(0))  # 返回数值部分
    else:
        return x


def train():
    pass


def main():
    weather = read_data_from_csv()
    weather.iloc[:, 1:] = weather.iloc[:, 1:].applymap(remove_unit)
    train()
