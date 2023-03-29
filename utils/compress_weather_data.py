import pandas as pd
import datetime as dt
import re
import os
import warnings

warnings.filterwarnings("ignore")

saved_weather_data = '../weather_data/pre-processed_weather_data.csv'


def generate_dataset():
    # divide the dataset into training and validation.
    # training set: from 2022-05-18 to 2023-03-08
    df_wtr = pd.read_csv('../data/electricity_data/pre-processed_weather_data.csv')[1::2]
    df_ele = pd.read_csv('../data/electricity_data/1A.csv')
    df_wtr['timestamp'] = pd.to_datetime(df_wtr['timestamp'])
    df_ele['time'] = pd.to_datetime(df_ele['time'])

    start_date = dt.datetime.strptime('2022-05-19', '%Y-%m-%d').date()
    end_date = dt.datetime.strptime('2023-03-08', '%Y-%m-%d').date()

    mask_wtr = (df_wtr['timestamp'].dt.date >= start_date) & (df_wtr['timestamp'].dt.date <= end_date)
    mask_ele = (df_ele['time'].dt.date >= start_date) & (df_ele['time'].dt.date <= end_date)
    weather_sub = df_wtr[mask_wtr][:-11]
    electricity_sub = df_ele[mask_ele]['val'][1:]
    training_dataset = pd.concat((weather_sub.reset_index(drop=True), electricity_sub.reset_index(drop=True)), axis=1)
    training_dataset.to_csv('../dataset/train_dataset/train.csv', index=False)

    # validation dataset: from 2021-02-05 to 2021-05-10
    start_date = dt.datetime.strptime('2021-02-05', '%Y-%m-%d').date()
    end_date = dt.datetime.strptime('2021-05-10', '%Y-%m-%d').date()
    mask_wtr = (df_wtr['timestamp'].dt.date >= start_date) & (df_wtr['timestamp'].dt.date <= end_date)
    mask_ele = (df_ele['time'].dt.date >= start_date) & (df_ele['time'].dt.date <= end_date)
    wtr_sub = df_wtr[mask_wtr][:-1]
    ele_sub = df_ele[mask_ele]['val'][1:]
    validation_dataset = pd.concat((wtr_sub.reset_index(drop=True), ele_sub.reset_index(drop=True)), axis=1)
    validation_dataset.to_csv('../dataset/validation_dataset/validation.csv', index=False)


def remove_unit(x):
    pattern = r"([-+]?\d*\.\d+|\d+)"  # 匹配数值部分的正则表达式
    match = re.search(pattern, x)  # 搜索匹配结果
    if match is not None:
        return float(match.group(0))  # 返回数值部分
    else:
        return x


def check_incomplete_date():
    std_shape = pd.read_csv('../weather_data/daily_weather_data/weather2020-07-31.csv').shape
    program_name = "logger"
    current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    folder_path = '../weather_data/daily_weather_data'
    logger = open(f"{program_name}.log", 'w')
    logger.write("\n")
    logger.close()
    logger = open(f"{program_name}.log", 'a')
    logger.write(f"The program '{program_name}' is running at {current_time}\n")
    weather_loss = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            csv_file = os.path.join(root, file)
            obj_file = pd.read_csv(csv_file)
            if obj_file.shape != std_shape:
                logger.write(f"inconsistent file:{file}, the corresponding shape is {obj_file.shape}\n")
                weather_loss.append(file[7:17])
    logger.close()
    return weather_loss


# remove the unit in each column and compress the daily weather data into one file
def compress_weather_data(incomplete_date):
    daily_folder = '../weather_data/daily_weather_data'
    weather_list = []

    for root, dirs, files in os.walk(daily_folder):
        for file in files:
            csv_file = os.path.join(root, file)
            weather = pd.read_csv(csv_file)
            # find the days when data are lost.
            date = file[7:17]
            if date in incomplete_date:
                continue
            weather.iloc[:, 2:] = weather.iloc[:, 2:].applymap(remove_unit)
            weather_list.append(weather)

        df = pd.concat(weather_list)
        df.to_csv(saved_weather_data, index=False)


# combine the time and date column into one timestamp column
def refine_timestamp():
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(saved_weather_data)

    # Combine date and time columns into a single datetime column
    df['datetime'] = df.apply(
        lambda x: dt.datetime.strptime(x['Date'] + ' ' + x['Time'], '%Y-%m-%d %I:%M %p') - dt.timedelta(hours=8),
        axis=1)
    # Format datetime column as a string in the desired format
    df['timestamp'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df = df.drop(['datetime', 'Date', 'Time'], axis=1)
    # move the 'timestamp' to first
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    df.to_csv(saved_weather_data, index=False)


if __name__ == "__main__":
    missing_date_weather = check_incomplete_date()
    compress_weather_data(missing_date_weather)
    refine_timestamp()
