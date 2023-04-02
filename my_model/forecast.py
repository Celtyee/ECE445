import json
import warnings
import pandas as pd
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

warnings.filterwarnings("ignore")


class deepAR_model:
    def __init__(self, pl_ckpt, context_length, predictor_length, building):
        self.model = DeepAR.load_from_checkpoint(pl_ckpt)
        self.building_series = building
        self.context_length = context_length
        self.predictor_length = predictor_length

    def predict(self, history_data):
        data = pd.read_csv(history_data)
        data = data.drop(['Wind', 'Precip.', 'Wind Gust'], axis=1)
        data = data.dropna()
        cutoff = data["time_idx"].max() - self.predictor_length

        print(f'The size of data is {len(data[lambda x: x["time_idx"] <= cutoff])}')

        print(f'The cutoff is {cutoff}')    # 359
        print(f'The max of time_idx is {data["time_idx"].max()}')   # 527
        print(f'The context length is {self.context_length}')   # 336
        print(f'The predictor length is {self.predictor_length}')   # 168

        history = TimeSeriesDataSet(
            data[lambda x: x.index <= cutoff],
            time_idx="time_idx",
            target="val",
            categorical_encoders={"Building": NaNLabelEncoder().fit(data.Building)},
            group_ids=["Building"],
            static_categoricals=["Building"],
            time_varying_known_reals=["Temperature", "Humidity", "Pressure"],
            allow_missing_timesteps=True,
            time_varying_unknown_reals=["val"],
            max_encoder_length=self.context_length,
            max_prediction_length=self.predictor_length
        )

        test = TimeSeriesDataSet.from_dataset(history, data, min_prediction_idx=cutoff + 1)
        batch_size = 128
        test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0,
                                             batch_sampler='synchronized')

        predictions = self.model.predict(test_dataloader)
        pred_dict = {}
        for idx in range(len(self.building_series)):
            pred_dict[self.building_series[idx]] = predictions[idx].tolist()

        with open("prediction.json", "w") as f:
            json.dump(pred_dict, f)
