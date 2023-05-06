import os.path
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from utils import train_api
import logging
import datetime
from generate_train_dataset_buildings import generate_train_dataset_buildings
import time


def self_train(data, hidden_size, rnn_layer, context_day, prediction_len, min_lr):
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
    batch_size = 1024

    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )
    model_name = f"hidden={hidden_size}-rnn_layer={rnn_layer}-context_day={context_day}-min_lr={min_lr}"
    save_folder_path = "my_model"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    trainer = train_api()
    net, ckpt_path = trainer.train_model(training, train_dataloader, val_dataloader, hidden_size, rnn_layer,
                                         model_name,
                                         min_lr, save_folder_path)
    loss = trainer.validation_model(net, save_folder_path, validation, val_dataloader, ckpt_path)
    return loss


def auto_train():
    hidden = 38
    rnn = 3
    context = 3
    prediction_len = 1

    pl_seed = 42
    pl.seed_everything(pl_seed)
    logger = logging.getLogger(f"train_auto")
    logging.basicConfig(filename=f'train_auto.txt',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO,
                        filemode='w')

    logger.critical(f"Training starts at {datetime.datetime.now()}")
    min_lr = 1e-3
    if not os.path.exists("my_model"):
        os.mkdir("my_model")

    train_dataset_path = generate_train_dataset_buildings()
    data = pd.read_csv(train_dataset_path)
    data = data.fillna(method="ffill")
    data = data.astype(dict(Building=str))

    # record the parameters
    logger.critical(
        f"hidden_size={hidden}, rnn_layers={rnn}, context_day={context}, prediction_len = {prediction_len}, min_lr={min_lr}, pl_seed = {pl_seed}")
    val_loss = self_train(data, hidden_size=hidden, rnn_layer=rnn, context_day=context, min_lr=min_lr,
                          prediction_len=prediction_len)
    logger.critical(f"loss = {val_loss}")

    logger.info("Train finish\n")


if __name__ == "__main__":
    # RECORD THE HYPERPARAMETERS: hidden=38-rnn_layer=3-context_day=3-min_lr=0.001
    auto_train()
    # sleep for 1 day
    time.sleep(86400)
