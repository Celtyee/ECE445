import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from pytorch_forecasting import DeepAR


# define the network and find the optimal learning rate for the specific task
def train_model(time_series_dataset, dataloader_train, dataloader_val, hidden_size=30, rnn_layers=2,
                save_folder="./", min_lr=1e-5):
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
        early_stop_threshold=100,

    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.savefig(f"{save_folder}/res.png")
    net.hparams.learning_rate = res.suggestion()

    # start train
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=50,
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

    trainer.save_checkpoint(f"{save_folder}/checkpoint.ckpt")
    return net


def validation_model(net, save_folder, timeseries_val, dataloader_val):
    model = DeepAR.load_from_checkpoint(f"{save_folder}/checkpoint.ckpt")
    actuals = torch.cat([y[0] for x, y in iter(dataloader_val)])
    predictions = model.predict(dataloader_val)
    loss = ((actuals - predictions).abs().mean())
    raw_predictions, x = net.predict(dataloader_val, mode="raw", return_x=True, n_samples=100)
    print(type(x), type(raw_predictions))
    series = timeseries_val.x_to_index(x)["Building"]
    for idx in range(len(series)):  # plot 10 examples
        model.plot_prediction(x, raw_predictions, idx=idx,
                              add_loss_to_title=True)
        building = series.iloc[idx]
        plt.suptitle(f"Building: {building}")
        plt.savefig(f"{save_folder}/plot_{building}_encoder.png")

    return loss
