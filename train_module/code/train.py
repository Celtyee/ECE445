import os.path

import pandas as pd
import pytorch_lightning as pl
import pytorch_forecasting as pf
import torch
from pytorch_forecasting import Baseline, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from utils import train_model, validation_model
import logging
import datetime
import argparse


def train(hidden_size, rnn_layer, context_day, min_lr):
    if not os.path.exists("../train_recorder"):
        os.mkdir("../train_recorder")

    data = pd.read_csv('../dataset/train/train.csv')
    data = data.drop(['Wind', 'Precip.', 'Wind Gust'], axis=1)
    data = data.dropna()
    data = data.astype(dict(Building=str))
    # create dataset and dataloaders
    max_encoder_length = int(24 * context_day)
    max_prediction_length = int(24 * 7)

    training_cutoff = data["time_idx"].max() - max_prediction_length
    # print(training_cutoff)

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.index <= training_cutoff],
        time_idx="time_idx",
        target="val",
        categorical_encoders={"Building": NaNLabelEncoder().fit(data.Building),
                              "Condition": NaNLabelEncoder().fit(data.Condition)},
        group_ids=["Building"],
        static_categoricals=[
            "Building"
        ],

        time_varying_known_reals=["Temperature", "Humidity"],
        time_varying_known_categoricals=["Condition"],
        allow_missing_timesteps=True,
        time_varying_unknown_reals=["val"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
    batch_size = 128

    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    # print(f"Number of batches: {len(val_dataloader)}")
    # print(f"Batch size: {val_dataloader.batch_size}")

    # # calculate baseline absolute error
    # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    # baseline_predictions = Baseline().predict(val_dataloader)
    # # baseline for being beat
    # print(SMAPE()(baseline_predictions, actuals))

    save_folder = f"../train_recorder/hidden={hidden_size}-rnn_layer={rnn_layer}-context_day={context_day}-min_lr={min_lr}"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    net = train_model(training, train_dataloader, val_dataloader, hidden_size, rnn_layer, save_folder, min_lr)
    loss = validation_model(net, save_folder, validation, val_dataloader)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, help="hidden size of deepAR network", default=30)
    parser.add_argument('--rnn_layers', type=int, help="number of rnn layers in deepAR network", default=2)
    parser.add_argument('--context_day', type=int, help="number of encoder", default=7)

    args = parser.parse_args()
    hidden = args.hidden_size
    rnn = args.rnn_layers
    context = args.context_day

    pl_seed = 42
    pl.seed_everything(pl_seed)
    logger = logging.getLogger("train_logger")
    logging.basicConfig(filename='log.txt',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO,
                        filemode='w')

    logger.info(f"Pytorch-version={torch.__version__}")
    logger.info(f"Pytorch-forecasting={pf.__version__}")
    logger.info(f"Pytorch-lightning={pl.__version__}")
    logger.critical(f"Training starts at {datetime.datetime.now()}")
    min_lr_list = [10 ** y for y in range(-4, 0)]
    for min_lr in min_lr_list:
        logger.critical(
            f"hidden_size={hidden}, rnn_layers={rnn}, context_day={context}, min_lr={min_lr}, pl_seed = {pl_seed}")
        val_loss = train(hidden_size=hidden, rnn_layer=rnn, context_day=context, min_lr=min_lr)
        logger.critical(f"loss = {val_loss}")

    logger.info("Train finish\n")
