import os.path

import pandas as pd
import pytorch_lightning as pl
import pytorch_forecasting as pf
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from utils import train_api
import logging
import datetime
import argparse


def train(data, hidden_size, rnn_layer, context_day, prediction_len, min_lr, task_name):
    # create dataset and dataloaders
    max_encoder_length = int(24 * context_day)
    max_prediction_length = int(24 * prediction_len)

    cutoff = data["time_idx"].max() - max_prediction_length
    # print(training_cutoff)

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.index <= cutoff],
        time_idx="time_idx",
        target="val",
        categorical_encoders={"Building": NaNLabelEncoder().fit(data.Building),
                              "Condition": NaNLabelEncoder().fit(data.Condition)},
        group_ids=["Building"],
        static_categoricals=[
            "Building"
        ],

        time_varying_known_reals=["Temperature", "Humidity", "is_weekend"],
        time_varying_known_categoricals=["Condition"],
        allow_missing_timesteps=True,
        time_varying_unknown_reals=["val"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=cutoff + 1)
    batch_size = 128

    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )
    model_name = f"hidden={hidden_size}-rnn_layer={rnn_layer}-context_day={context_day}-min_lr={min_lr}"
    save_folder_path = f"../data/train/{task_name}/{model_name}"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    trainer = train_api()
    net, ckpt_path = trainer.train_model(training, train_dataloader, val_dataloader, hidden_size, rnn_layer,
                                         model_name,
                                         min_lr, save_folder_path)
    loss = trainer.validation_model(net, save_folder_path, validation, val_dataloader, ckpt_path)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, help="hidden size of deepAR network")
    parser.add_argument('--rnn_layers', type=int, help="number of rnn layers in deepAR network")
    parser.add_argument('--context_day', type=int, help="number of encoder")
    parser.add_argument('--prediction_len', type=int, help="number of prediction")
    parser.add_argument('--task_name', type=str, help="the name of task")

    args = parser.parse_args()
    hidden = args.hidden_size
    rnn = args.rnn_layers
    context = args.context_day

    pl_seed = 42
    pl.seed_everything(pl_seed)
    logger = logging.getLogger("train_logger")
    logging.basicConfig(filename='train_logger.txt',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO,
                        filemode='w')

    logger.info(f"Pytorch-version={torch.__version__}")
    logger.info(f"Pytorch-forecasting={pf.__version__}")
    logger.info(f"Pytorch-lightning={pl.__version__}")
    logger.critical(f"Training starts at {datetime.datetime.now()}")
    min_lr_list = [10 ** y for y in range(-4, -3)]

    if not os.path.exists(f"../data/train/{args.task_name}"):
        os.mkdir(f"../data/train/{args.task_name}")

    train_dataset_path = "../data/train/train_buildings.csv"
    data = pd.read_csv(train_dataset_path)
    data = data.fillna(method="ffill")
    data = data.astype(dict(Building=str))

    for min_lr in min_lr_list:
        # record the hyperparameters
        logger.critical(
            f"hidden_size={hidden}, rnn_layers={rnn}, context_day={context}, prediction_len = {args.prediction_len}, min_lr={min_lr}, pl_seed = {pl_seed}, task_name = {args.task_name}")
        val_loss = train(data, hidden_size=hidden, rnn_layer=rnn, context_day=context, min_lr=min_lr,
                         task_name=args.task_name, prediction_len=args.prediction_len)
        logger.critical(f"loss = {val_loss}")

    logger.info("Train finish\n")


if __name__ == "__main__":
    main()
