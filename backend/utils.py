import numpy as np

scaler_min = 0
scaler_max = 240.0


class MinMaxScaler:
    def __init__(self, scaler_min: float, scaler_max: float):
        self.scaler_min = scaler_min
        self.scaler_max = scaler_max

    def transform(self, input_sequence: np.ndarray):
        return (input_sequence - self.scaler_min) / (self.scaler_max - self.scaler_min)

    def inverse_transform(self, input_sequence: np.float32):
        return (input_sequence * (self.scaler_max - self.scaler_min)) + self.scaler_min
