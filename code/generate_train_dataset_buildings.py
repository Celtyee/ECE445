from utils import dataset_generator
import datetime
import pandas as pd


def main():
    weather_path = "../data/weather"
    electricity_path = "../data/electricity"
    whole_weather_path = "../data/weather/history/pre-processed_weather.csv"

    generator = dataset_generator(weather_path, electricity_path)
    # generate the dataset from 20221015 to 20230308
    buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']
    start_date = datetime.datetime.strptime("20221101", "%Y%m%d").date()
    end_date = datetime.datetime.strptime("20230308", "%Y%m%d").date()
    train_df_list = generator.generate_dataset(buildings, start_date, end_date, whole_weather_path, start_idx=1,
                                               weather_stride=2)
    train_df = pd.concat(train_df_list)
    train_path = "../data/train/train_buildings.csv"
    train_df = train_df.fillna(method="ffill")
    train_df.to_csv(train_path, index=False)

    return None


if __name__ == "__main__":
    main()
