import torch
import datetime
import re
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
import logging

warnings.filterwarnings("ignore")


def remove_unit(x):
    pattern = r"([-+]?\d*\.\d+|\d+)"  # 匹配数值部分的正则表达式
    match = re.search(pattern, x)  # 搜索匹配结果
    if match is not None:
        return float(match.group(0))  # 返回数值部分
    else:
        return x


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

    def generate_dataset(self, building_list, start_date, end_date, weather, start_idx, weather_stride):
        '''
        generate the electricity usage dataset consists of weather condition, building ID, electricity usage.

        Parameters
        ----------
        building_list: list of strings
        start_date: datetime.datetime.date() data
        end_date: datetime.datetime.date() data
        weather: compressed weather data file path, str.
        start_idx: The idx of valid weather start, int.
        weather_stride: stride to the weather, int.
        Returns
        -------

        '''
        name_list = building_list
        df_list = []
        # generate the weather data for 10 buildings
        df_wtr = pd.read_csv(weather)[start_idx::weather_stride]
        df_wtr['timestamp'] = pd.to_datetime(df_wtr['timestamp'])

        mask_wtr = (df_wtr['timestamp'].dt.date >= start_date) & (df_wtr['timestamp'].dt.date <= end_date)
        weather_sub = df_wtr[mask_wtr][:]
        # only choose the column "timestamp", "Temperature", "Humidity", "Conditions"
        weather_sub = weather_sub[['timestamp', 'Temperature', 'Humidity', 'Condition']]

        for building in name_list:
            electricity_path = f'{self.history_electricity_folder}/{building}_complete.csv'
            df_ele = pd.read_csv(electricity_path)
            # change the name of timestamp into time
            df_ele.rename(columns={'timestamp': 'time'}, inplace=True)
            # Turn from UTC into UTC+8. -1 to align the val so that each values represent the electricity consumption in the next data.
            df_ele['time'] = pd.to_datetime(df_ele['time']) + datetime.timedelta(hours=8) - datetime.timedelta(hours=1)

            mask_ele = (df_ele['time'].dt.date >= start_date) & (df_ele['time'].dt.date <= end_date)
            # print(f"the number of losing electricity data for building {building} is {np.sum(val_mask)}")
            electricity_sub = df_ele[mask_ele]['val']
            training_df = pd.concat((weather_sub.reset_index(drop=True), electricity_sub.reset_index(drop=True)),
                                    axis=1)

            training_df['Building'] = building
            training_df['time_idx'] = range(len(training_df))
            # the 'val' is the electricity consumption between 'time_index' and 'time_index' + 1hour
            # add whether this hour is holiday or weekend
            # print(training_df.columns)
            # "is_weekend" is true if the day is weekend and the day is in summer holiday or winter holiday
            training_df['is_weekend'] = training_df['timestamp'].dt.dayofweek.isin([5, 6])
            # check whether the day is between June and September or between January and February
            training_df['is_holiday'] = ((training_df['timestamp'].dt.month >= 6) & (
                    training_df['timestamp'].dt.month <= 8)) | ((training_df['timestamp'].dt.month >= 1) & (
                    training_df['timestamp'].dt.month <= 2))

            df_list.append(training_df)

        return df_list

    def __check_incomplete_date(self):
        std_shape = (48, 11)
        # create a logger and record the incomplete date using logging
        logger = logging.getLogger("incomplete_date_logger")
        logging.basicConfig(filename='incomplete_date_logger.txt',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.INFO)

        folder_path = self.daily_weather_folder
        incomplete_weather = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                csv_file = os.path.join(root, file)
                obj_file = pd.read_csv(csv_file)
                if obj_file.shape != std_shape:
                    log_sentence = f"inconsistent file:{file}, the corresponding shape is {obj_file.shape}\n"
                    # print(log_sentence)
                    logger.critical(log_sentence)
                    incomplete_weather.append(file[7:17])
        return incomplete_weather

    def compress_weather_data(self, total_csv_save_path):
        '''
        Compress the all daily weather data into one file, removing the units.

        Parameters
        ----------
        total_csv_save_path

        Returns
        -------

        '''
        incomplete_date = self.__check_incomplete_date()
        daily_folder = self.daily_weather_folder
        weather_list = []
        for root, dirs, files in os.walk(daily_folder):
            for file in sorted(files):
                # print(f"Now we are processing csv file {file}")
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


# define the network and find the optimal learning rate for the specific task
class train_api:
    def train_model(self, time_series_dataset, dataloader_train, dataloader_val, hidden_size, rnn_layers,
                    model_name, min_lr, save_folder_path, epochs_number=200, daily_train=False):
        trainer = pl.Trainer(gpus=1, gradient_clip_val=1e-1)
        net = DeepAR.from_dataset(
            time_series_dataset, learning_rate=3e-2, hidden_size=hidden_size, rnn_layers=rnn_layers
        )

        # find optimal learning rate
        res = trainer.tuner.lr_find(
            net,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val,
            min_lr=min_lr,
            max_lr=1e0,
            early_stop_threshold=100
        )
        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=False, suggest=True)
        fig.savefig(f"{save_folder_path}/res.png")
        net.hparams.learning_rate = res.suggestion()

        # start train
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=epochs_number,
            gpus=1,
            enable_model_summary=True,
            gradient_clip_val=1e-2,
            callbacks=[early_stop_callback],
            limit_train_batches=50,
            enable_checkpointing=True,
        )

        trainer.fit(
            net,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_val,
        )
        ckpt_path = f"{save_folder_path}/{model_name}.ckpt"
        trainer.save_checkpoint(ckpt_path)
        if daily_train:
            daily_path = f"../data/train/daily_train/{model_name}"
            trainer.save_checkpoint(f"{daily_path}/{model_name}.ckpt")

        return net, ckpt_path

    def validation_model(self, net, save_folder_path, timeseries_val, dataloader_val, ckpt_path):
        model = DeepAR.load_from_checkpoint(ckpt_path)
        actuals = torch.cat([y[0] for x, y in iter(dataloader_val)])
        predictions = model.predict(dataloader_val)
        loss = ((actuals - predictions).abs().mean())
        raw_predictions, x = net.predict(dataloader_val, mode="raw", return_x=True, n_samples=100)
        # print(type(x), type(raw_predictions))
        series = timeseries_val.x_to_index(x)["Building"]
        for idx in range(len(series)):  # plot 10 examples
            model.plot_prediction(x, raw_predictions, idx=idx,
                                  add_loss_to_title=True)
            building = series.iloc[idx]
            plt.suptitle(f"Building: {building}")
            plt.savefig(f"{save_folder_path}/plot_{building}_encoder.png")

        return loss


class my_deepAR_model:
    def __init__(self, pl_ckpt, context_length, prediction_len, building):
        self.model = DeepAR.load_from_checkpoint(pl_ckpt)
        self.building_series = building
        self.context_length = context_length
        self.prediction_len = prediction_len
        print(self.prediction_len)

    # NOTE change this will affect the test program.
    def predict(self, csv_data_path):
        '''
        Predict the future electricity usage with the prediction data.
        Parameters
        ----------
        csv_data_path: The path of necessary data for prediction, including historical weather and electricity usage data
                        as well as the future weather data, str.

        Returns
        -------

        '''
        data = pd.read_csv(csv_data_path)
        cutoff = data["time_idx"].max() - self.prediction_len

        # NOTE: features needs to be added or removed if the features of training set is changed.
        history = TimeSeriesDataSet(
            data[lambda x: x.index <= cutoff],
            time_idx="time_idx",
            target="val",
            categorical_encoders={"Building": NaNLabelEncoder().fit(data.Building),
                                  "Condition": NaNLabelEncoder().fit(data.Condition)},
            group_ids=["Building"],
            static_categoricals=[
                "Building"
            ],

            time_varying_known_reals=["Temperature", "Humidity", "is_weekend", "is_holiday"],
            time_varying_known_categoricals=["Condition"],
            allow_missing_timesteps=True,
            time_varying_unknown_reals=["val"],
            max_encoder_length=self.context_length,
            max_prediction_length=self.prediction_len
        )

        test = TimeSeriesDataSet.from_dataset(history, data, min_prediction_idx=cutoff + 1)
        batch_size = 128
        test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0,
                                             batch_sampler='synchronized')

        predictions = self.model.predict(test_dataloader)
        pred_dict = {}
        for idx in range(len(self.building_series)):
            pred_dict[self.building_series[idx]] = predictions[idx].tolist()

        return pred_dict

    def rollback_predict(self, input_data_path):
        '''
        Predict the future electricity usage with the prediction data.
        Parameters
        ----------
        input_data_path: The path of necessary data for prediction, including historical weather and electricity usage data
                        as well as the future weather data, str.

        Returns
        -------

        '''
        data = pd.read_csv(input_data_path)
        cutoff = data["time_idx"].max() - self.prediction_len

        num_days = int(self.prediction_len / 24)
        pred_dict = {}
        for d in range(num_days):
            context_idx = cutoff + d * 24
            # NOTE: features needs to be added or removed if the features of training set is changed.
            context = TimeSeriesDataSet(
                data[lambda x: x.index <= context_idx],
                time_idx="time_idx",
                target="val",
                categorical_encoders={"Building": NaNLabelEncoder().fit(data.Building),
                                      "Condition": NaNLabelEncoder().fit(data.Condition)},
                group_ids=["Building"],
                static_categoricals=[
                    "Building"
                ],

                time_varying_known_reals=["Temperature", "Humidity", "is_weekend", "is_holiday"],
                time_varying_known_categoricals=["Condition"],
                allow_missing_timesteps=True,
                time_varying_unknown_reals=["val"],
                max_encoder_length=self.context_length,
                max_prediction_length=24
            )
            # print(self.predictor_length)
            batch_size = 128
            prediction_start_idx = context_idx + 1
            print(f"day: {d}")
            prediction_data = TimeSeriesDataSet.from_dataset(context, data, min_prediction_idx=prediction_start_idx)
            pred_dataloader = prediction_data.to_dataloader(train=False, batch_size=batch_size, num_workers=0,
                                                            batch_sampler='synchronized')

            predictions = self.model.predict(pred_dataloader)
            for idx in range(len(self.building_series)):
                prediction_list = predictions[idx].tolist()
                building = self.building_series[idx]
                if sum(prediction_list) < 24:
                    context_2day_ago = np.array(data[prediction_start_idx - 48:prediction_start_idx - 24]["val"])
                    context_1day_ago = np.array(data[prediction_start_idx - 24:prediction_start_idx]["val"])
                    #     get the average of two list
                    prediction_list = (context_2day_ago + context_1day_ago) / 2
                    prediction_list = prediction_list.tolist()
                    # if self.building_series[idx] == '1D':
                    #     print(f"trigger for building 1D at day {d + 1}")
                    #     print(prediction_list)
                pred_dict[self.building_series[idx]] = pred_dict.get(self.building_series[idx], []) + prediction_list
            # FIXME: attach the prediction result to the corresponding building
            data.to_csv(f"day_{d + 1}_rollback.csv",
                        index=False)  # NOTE: to be commented out if the data is not needed to be saved.
        return pred_dict


class electricity_complete_api:
    '''
    fulfill the electricity usage data if the data is incomplete in 24 hours for a day.
    '''

    def fetch_blank_data(self, electricity_sub, electricity_sub_date):
        '''

        Parameters
        ----------
        electricity_sub: the electricity data for a specific date, pandas.dataframe.
        electricity_sub_date: the date of the data, datetime.date.

        Returns
        -------
        the data frame with timestamp that from 0:00 to 23:00 hourly in one day. The lack data value is 0.
        '''
        # the 0:00 of the date, timezone = UTC.
        # NOTE: Do not use the min time of the date because the min time is not always 0:00 for a input df.
        zero_time = datetime.datetime.combine(electricity_sub_date, datetime.time(0, 0)).replace(
            tzinfo=datetime.timezone.utc)
        # create a data frame with timestamp that has 24 hours in that date
        fulfill_df = pd.DataFrame(columns=electricity_sub.columns)
        for h in range(24):
            fulfill_df = fulfill_df.append(
                {'time': zero_time + datetime.timedelta(hours=h), 'val': 0},
                ignore_index=True)
        for idx in range(len(electricity_sub)):
            # get the number of hours between the 0:00 of the date and the time of electricity_sub
            # turn the datatime into no timezones
            exist_hour = electricity_sub['time'].iloc[idx]
            hours_diff = (exist_hour - zero_time).total_seconds() / 3600
            fulfill_df.iloc[int(hours_diff)] = electricity_sub.iloc[idx]
        # electricity_sub.to_csv("electricity_sub.csv", index=False)
        # fufill_df.to_csv("fufill_df.csv", index=False)
        return fulfill_df

    def complete_electricity_total(self, buildings=None):
        '''
        Complete all the electricity usage data.
        Parameters
        ----------
        buildings

        Returns
        -------

        '''
        if buildings is None:
            buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']
        for building in buildings:
            print(building)
            df = pd.read_csv(f"../data/electricity/{building}.csv")
            df['time'] = pd.to_datetime(df['time']) + datetime.timedelta(hours=7)
            # data range from the first date to last date in the data frame
            start_date = df['time'].min().date()
            end_date = df['time'].max().date()
            print(f"{start_date}, {end_date}")
            # for each date in the data frame
            for date in pd.date_range(start_date, end_date):
                # get the electricity data for the specific date
                electricity_sub = df[df['time'].dt.date == date]
                # if the electricity data is incomplete, fulfill the data
                if len(electricity_sub['time']) < 24:
                    # print(f"incomplete date for date {date}")
                    fulfill_df = self.fetch_blank_data(electricity_sub, date)
                    # remove the electricity_sub from the whole data frame
                    df = df[df['time'].dt.date != date]
                    df = df.append(fulfill_df)
                # sort the data frame by time
            df = df.sort_values(by=['time'])
            df['time'] = pd.to_datetime(df['time']) - datetime.timedelta(hours=7)
            # NOTE: set the upper bound of the electricity as 500
            # set the 'val' column of the dataframe to nan if it is less than 0 or greater than 1000
            val_mask = (df['val'] <= 0.0) | (df['val'] >= 500.0)
            df.loc[val_mask, 'val'] = np.nan
            # fill the nan with the mean of previous 48 data points
            for i in range(48, len(df)):
                if not np.isnan(df['val'].iloc[i]):
                    continue
                df['val'].iloc[i] = np.mean(df['val'].iloc[i - 48:i])
            df.to_csv(f"../data/electricity/{building}_complete.csv", index=False)
