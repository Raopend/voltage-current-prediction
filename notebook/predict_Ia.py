import warnings

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


class UnivariateSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: pd.DataFrame,
        n_lags: int,
        horizon: int,
        test_size: float = 0.2,
        batch_size: int = 16,
        feature_name: str = "",
    ) -> None:
        super().__init__()
        self.data = data
        self.n_lags = n_lags
        self.horizon = horizon
        self.test_size = test_size
        self.batch_size = batch_size
        self.feature_name = feature_name
        self.training: TimeSeriesDataSet
        self.validation: TimeSeriesDataSet
        self.test: TimeSeriesDataSet
        self.predict: TimeSeriesDataSet
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.setup()

    def preprocess_data(self):
        self.data["target"] = self.data[self.feature_name]
        self.data["time_index"] = np.arange(len(self.data))
        self.data["group_id"] = 0

    def split_data(self):
        time_indices = self.data["time_index"].values
        train_indices, test_indices = train_test_split(
            time_indices, test_size=self.test_size, shuffle=False
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.1, shuffle=False
        )
        return train_indices, val_indices, test_indices

    def scale_target(self, df, indices):
        scaled_values = self.target_scaler.transform(df.loc[indices, ["target"]])
        df.loc[indices, "target"] = scaled_values

    def setup(self, stage=None) -> None:
        self.preprocess_data()
        train_indices, val_indices, test_indices = self.split_data()

        train_df = self.data.loc[self.data["time_index"].isin(train_indices)]
        val_df = self.data.loc[self.data["time_index"].isin(val_indices)]
        test_df = self.data.loc[self.data["time_index"].isin(test_indices)]
        # predict_df = pd.concat([train_df, val_df, test_df])

        self.target_scaler.fit(train_df[["target"]])
        self.scale_target(train_df, train_df.index)
        self.scale_target(val_df, val_df.index)
        self.scale_target(test_df, test_df.index)

        self.training = TimeSeriesDataSet(
            train_df,
            time_idx="time_index",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            time_varying_unknown_reals=[self.feature_name],
            scalers={
                name: MinMaxScaler(feature_range=(0, 1)) for name in self.feature_name
            },
        )
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, val_df, stop_randomization=True
        )
        self.test = TimeSeriesDataSet.from_dataset(
            self.training, test_df, stop_randomization=True
        )
        self.predict = TimeSeriesDataSet.from_dataset(
            self.training, test_df, stop_randomization=True
        )

    def train_dataloader(self):
        return self.training.to_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.validation.to_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.test.to_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.predict.to_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
        )


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        loss = F.mse_loss(y_pred, y[0])
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        loss = F.mse_loss(y_pred, y[0])
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        loss = F.mse_loss(y_pred, y[0])
        self.log("validation_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.detach().numpy()
        y_pred = self.trainer.datamodule.target_scaler.inverse_transform(y_pred)
        return y_pred

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimiser


if __name__ == "__main__":
    # 选取一天内数据观察
    df = pd.read_csv(
        "../data/dnaq_15s_interval.csv",
        parse_dates=["collection_time"],
        index_col="collection_time",
    )
    df = df.loc["2022-01-01":"2022-01-01"]
    datamodule = UnivariateSeriesDataModule(
        df, n_lags=5, horizon=5, feature_name="Ia", batch_size=32
    )
    model = LSTMModel(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
    pl.seed_everything(42)
    logger = TensorBoardLogger("lightning_logs", name="voltage-prediction")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        accelerator="cpu",
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
