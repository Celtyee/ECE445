import os.path
import pandas as pd
import pytorch_lightning as pl
import logging
import datetime
from generate_train_dataset_buildings import generate_train_dataset_buildings
import time
from fetch_weather import fetch_weather_daily
from merge_electricity_data import merge_electricity_oneday
from test_module import test
from train import train
from shutil import copyfile

logger = logging.getLogger('daily_train')
task_name = "daily_train"
save_folder = f"../data/train/{task_name}"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)


def daily_train(hidden_size, rnn_layers, context_len, prediction_len):
    # hyper-parameters for training
    pl_seed = 42
    pl.seed_everything(pl_seed)
    logging.basicConfig(filename='daily_train.log',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO,
                        filemode='w')

    logger.critical(f"Training starts at {datetime.datetime.now()}")
    min_lr = 1e-3
    if not os.path.exists("my_model"):
        os.mkdir("my_model")

    train_dataset_path = generate_train_dataset_buildings(daily_train=True, logger=logger)
    data = pd.read_csv(train_dataset_path)
    data = data.fillna(method="ffill")
    data = data.astype(dict(Building=str))

    # record the parameters
    logger.critical(
        f"hidden_size={hidden_size}, rnn_layers={rnn_layers}, context_day={context_len}, prediction_len = {prediction_len}, min_lr={min_lr}, pl_seed = {pl_seed}")
    val_loss = train(data, hidden_size=hidden_size, rnn_layer=rnn_layers, context_day=context_len, min_lr=min_lr,
                     prediction_len=prediction_len, task_name=task_name)
    logger.critical(f"loss = {val_loss}")

    logger.info("Train finish\n")

    model_name = f"hidden={hidden_size}-rnn_layer={rnn_layers}-context_day={context_len}-min_lr={min_lr}"
    return model_name


if __name__ == "__main__":
    hidden_size = 38
    rnn_layers = 3
    context_len = 7
    # NOTE: change for training process
    prediction_len = 7
    fetch_date = datetime.datetime.today() - datetime.timedelta(days=1)
    fetch_weather_daily(fetch_date.date())
    merge_electricity_oneday(fetch_date.date())
    model_name = daily_train(hidden_size, rnn_layers, context_len, prediction_len)
    trained_model_path = f"../data/train/{task_name}/{model_name}/{model_name}.ckpt"
    print(trained_model_path)
    if not os.path.exists("my_model"):
        os.mkdir("my_model")
    # copy the trained model back to my model folder
    copyfile(trained_model_path, f"my_model/{model_name}.ckpt")
    # NOTE: change for testing process
    test(model_name=model_name, task_name=task_name, prediction_len=7, rollback=False)
    # sleep for 1 day
    time.sleep(86400)
