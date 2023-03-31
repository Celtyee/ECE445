import pandas as pd
import datetime
import os
import warnings
import re

warnings.filterwarnings("ignore")


def remove_unit(x):
    pattern = r"([-+]?\d*\.\d+|\d+)"  # 匹配数值部分的正则表达式
    match = re.search(pattern, x)  # 搜索匹配结果
    if match is not None:
        return float(match.group(0))  # 返回数值部分
    else:
        return x


def check_incomplete_electricity_dt(electricity_csv):
    # Load the data into a pandas dataframe
    df = pd.read_csv(electricity_csv)

    # Convert the timestamp column to datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Set the timestamp column as the index of the dataframe
    df.set_index('time', inplace=True)

    # Resample the dataframe to hourly frequency
    df_resampled = df.resample('H').mean()

    # Get the list of dates where the resampled data is missing
    missing_dates = df_resampled[df_resampled.isna().any(axis=1)].index.strftime('%Y-%m-%d').unique().tolist()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    program_name = "logger"
    logger = open(f"{program_name}.log", 'w')
    logger.write(f"\n\n\nThe program 'check_incomplete_electricity_dt' is running at {current_time}\n")
    # Print the missing dates
    logger.write("Dates with non-hourly data:\n")

    count = 0
    for date in missing_dates:
        if count % 5 == 0:
            logger.write("\n")
        logger.write(f" {str(date)} ")
        count += 1


# generator the dataset containing weather condition and
def refine_timestamp(total_csv_save_path):
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(total_csv_save_path)

    # Combine date and time columns into a single datetime column
    df['datetime'] = df.apply(
        lambda x: datetime.datetime.strptime(x['Date'] + ' ' + x['Time'], '%Y-%m-%d %I:%M %p'), axis=1)
    # Format datetime column as a string in the desired format
    df['timestamp'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df = df.drop(['datetime', 'Date', 'Time'], axis=1)
    # move the 'timestamp' to first
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    df.to_csv(total_csv_save_path, index=False)


class dataset_generator:
    def __init__(self, weather, electricity):
        self.daily_weather_folder = f'{weather}/daily_weather'
        self.history_weather_folder = weather
        self.history_electricity_folder = electricity

    def generate_dataset(self, building_list, start_date, end_date, weather):
        name_list = building_list
        df_list = []
        # generate the weather data for 10 buildings
        df_wtr = pd.read_csv(weather)[1::2]
        df_wtr['timestamp'] = pd.to_datetime(df_wtr['timestamp'])

        mask_wtr = (df_wtr['timestamp'].dt.date >= start_date) & (df_wtr['timestamp'].dt.date <= end_date)
        weather_sub = df_wtr[mask_wtr][:]

        for building in name_list:
            df_ele = pd.read_csv(f'{self.history_electricity_folder}/{building}.csv')
            df_ele['time'] = pd.to_datetime(df_ele['time']) + datetime.timedelta(hours=8)

            mask_ele = (df_ele['time'].dt.date >= start_date) & (df_ele['time'].dt.date <= end_date)

            electricity_sub = df_ele[mask_ele]['val'][1:]
            training_dataset = pd.concat((weather_sub.reset_index(drop=True), electricity_sub.reset_index(drop=True)),
                                         axis=1)

            training_dataset['Building'] = building
            training_dataset['time_idx'] = range(len(training_dataset))
            # the 'val' is the electricity consumption between 'time_index' and 'time_index' + 1hour
            df_list.append(training_dataset)

        return df_list

    def __check_incomplete_date(self):
        std_shape = (48, 11)
        program_name = "logger"
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        folder_path = self.daily_weather_folder
        logger = open(f"{program_name}.log", 'w')
        logger.write("\n")
        logger.close()
        logger = open(f"{program_name}.log", 'a')
        logger.write(f"The program '{program_name}' is running at {current_time}\n")
        incomplete_weather = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                csv_file = os.path.join(root, file)
                obj_file = pd.read_csv(csv_file)
                if obj_file.shape != std_shape:
                    logger.write(f"inconsistent file:{file}, the corresponding shape is {obj_file.shape}\n")
                    incomplete_weather.append(file[7:17])
        logger.close()
        return incomplete_weather

    # remove the unit in each column and compress the daily weather data into one file
    # TODO: function reach: only compress the data that is complete and in a specific date range.
    def compress_weather_data(self, total_csv_save_path):
        incomplete_date = self.__check_incomplete_date()
        daily_folder = self.daily_weather_folder
        weather_list = []
        for root, dirs, files in os.walk(daily_folder):
            for file in files:
                csv_file = os.path.join(root, file)
                weather = pd.read_csv(csv_file)
                # # find the days when data are lost.
                date = file[7:17]
                if date in incomplete_date:
                    continue
                weather.iloc[:, 2:] = weather.iloc[:, 2:].applymap(remove_unit)
                weather_list.append(weather)

        weather_df = pd.concat(weather_list)
        weather_df.to_csv(total_csv_save_path, index=False)
        refine_timestamp(total_csv_save_path)
