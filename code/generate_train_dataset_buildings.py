import os.path

from utils import dataset_generator
import datetime
import pandas as pd


def generate_train_dataset_buildings(daily_train=False):
    train_path = "../data/train"
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    weather_path = "../data/weather"
    electricity_path = "../data/electricity"
    whole_weather_path = "../data/weather/history/history_weather_vc.csv"

    generator = dataset_generator(weather_path, electricity_path)
    buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']
    start_date = datetime.datetime.strptime("20220101", "%Y%m%d").date()
    if daily_train:
        end_date = datetime.datetime.today().date() - datetime.timedelta(days=8)
    else:
        end_date = datetime.datetime.strptime("20230430", "%Y%m%d").date()
    print(f"The training set is from {start_date} to {end_date}.\n")
    train_df_list = generator.generate_dataset(buildings, start_date, end_date, whole_weather_path, start_idx=1,
                                               weather_stride=1)
    drop_date_start = datetime.datetime.strptime("20220615", "%Y%m%d").date()
    drop_date_end = datetime.datetime.strptime("20221015", "%Y%m%d").date()
    for i in range(len(train_df_list)):
        if i == 0:
            print(train_df_list[i].columns)
        # drop the rows from 20220615 to 20221015
        ele_mask = (train_df_list[i]['timestamp'].dt.date >= drop_date_start) & (
                train_df_list[i]['timestamp'].dt.date <= drop_date_end)
        train_df_list[i] = train_df_list[i].loc[~ele_mask]
        train_df_list[i]['time_idx'] = range(len(train_df_list[i]))
    train_df = pd.concat(train_df_list)
    train_path = "../data/train/train_buildings.csv"
    # pre-process for the weather information.
    # train_df = train_df.fillna(method="ffill")
    train_df.to_csv(train_path, index=False)

    return train_path


if __name__ == "__main__":
    generate_train_dataset_buildings()
