from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models import DeepAR

# load data
# assuming you have loaded and preprocessed your data into a pandas DataFrame with columns 'target', 'time_idx', and 'feat_dynamic_real'
data = ...

# create dataset
max_encoder_length = 60
max_prediction_length = 30
training_cutoff = data["time_idx"].max() - max_prediction_length
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["feat_dynamic_real"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)

# create dataloader
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

# create model
model = DeepAR.from_dataset(
    training,
    learning_rate=1e-3,
    log_interval=10,
    log_val_interval=1,
    weight_decay=1e-2,
    use_embedding=True,
    embedding_dimension=16,
)

# fit model
num_epochs = 100
trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, gradient_clip_val=0.1)
trainer.fit(model, train_dataloader)

# make predictions
new_prediction_data = data[lambda x: x.time_idx > training_cutoff]
new_prediction_dataset = TimeSeriesDataSet(
    new_prediction_data,
    time_idx="time_idx",
    target="target",
    group_ids=["feat_dynamic_real"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)
new_prediction_dataloader = new_prediction_dataset.to_dataloader(batch_size=batch_size, num_workers=0)
predictions = model.predict(new_prediction_dataloader)
