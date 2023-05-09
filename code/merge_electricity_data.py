import os.path

import numpy as np

from utils import electricity_complete_api
import pandas as pd
import datetime


def merge_save_electricity():
    buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']
    save_electricity_path = "../DataCollector/target/file/2023.03.08+2023-04-24.csv"
    new_electricity_data = pd.read_csv(save_electricity_path)
    for building in buildings:
        df_building = pd.read_csv(f"../data/electricity/{building}.csv")
        building_metric = df_building['metric'].unique()
        df_building_new = new_electricity_data[new_electricity_data['metric'].isin(building_metric)]
        start_date = datetime.datetime.strptime("20230308", "%Y%m%d").date()
        df_building['time'] = pd.to_datetime(df_building['time'])
        df_old = df_building[df_building['time'].dt.date < start_date]

        df_building_new['time'] = pd.to_datetime(df_building_new['time'])
        df_new = df_building_new[df_building_new['time'].dt.date >= start_date]
        # set timezone as UTC
        df_new['time'] = df_new['time'].dt.tz_localize('UTC')

        df_building_complete = pd.concat([df_old, df_new])
        df_building_complete.to_csv(f"../data/electricity/{building}.csv", index=False)

    print("complete the electricity data")
    completeAPI = electricity_complete_api()
    completeAPI.complete_electricity_total()


def merge_electricity_oneday(input_date):
    '''

    Parameters
    ----------
    input_date: datetime.datetime.date(). The date of the new electricity data.

    Returns
    -------

    '''
    buildings = ['1A', '1B', '1C', '1D', '1E', '2A', '2B', '2C', '2D', '2E']
    select_date = input_date
    completeAPI = electricity_complete_api()
    for building in buildings:
        csv_name = f"v.building.A00{building}.elec.hourly+{select_date}.csv"
        print(csv_name)
        csv_path = f"../DataCollector/target/file/{csv_name}"
        # if not os.path.exists(csv_path):
        #     continue
        df_building_new = pd.read_csv(csv_path)
        df_building_new['time'] = pd.to_datetime(df_building_new['time']).dt.tz_localize('UTC')
        df_building_new['time'] += datetime.timedelta(hours=17)
        df_building_new = completeAPI.fetch_blank_data(df_building_new, select_date)
        df_building_new['time'] -= datetime.timedelta(hours=17)
        val_mask = (df_building_new['val'] <= 0.0) | (df_building_new['val'] >= 500.0)
        df_building_new.loc[val_mask, 'val'] = np.nan

        start_time = df_building_new['time'].min()
        # print(start_time)
        df_building_old = pd.read_csv(f"../data/electricity/{building}_complete.csv")
        df_building_old['time'] = pd.to_datetime(df_building_old['time'])
        df_building_old = df_building_old[df_building_old['time'] < start_time]

        df_building_complete = pd.concat([df_building_old, df_building_new])
        # print(len(df_building_old), len(df_building_complete))
        for i in range(len(df_building_old), len(df_building_complete)):
            if not np.isnan(df_building_complete['val'].iloc[i]):
                continue
            df_building_complete['val'].iloc[i] = np.mean(df_building_complete['val'].iloc[i - 48:i])
        df_building_complete.to_csv(f"../data/electricity/{building}_complete.csv", index=False)


if __name__ == "__main__":
    # merge_save_electricity()
    date_end = (datetime.datetime.today() - datetime.timedelta(days=1)).date()
    date_start = date_end - datetime.timedelta(days=6)
    for date in pd.date_range(date_start, date_end):
        print(date)
        merge_electricity_oneday(date.date())
