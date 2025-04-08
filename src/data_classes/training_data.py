from dataclasses import dataclass
import os
import pandas as pd
import torch
from torch import Tensor
from .enums import SamplingMethod

@dataclass
class TrainingData:
    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor

@dataclass
class InputData:
    sobol_x: Tensor
    sobol_y: Tensor
    mc_x: Tensor
    mc_y: Tensor

    def __init__(self, data_path: str):
        sobol_data = pd.read_csv(os.path.join(data_path, 'input', 'sobol_sample.csv'))
        mc_data = pd.read_csv(os.path.join(data_path, 'input', 'mc_sample.csv'))

        self.sobol_x = torch.tensor(sobol_data.iloc[:, :-1].values, dtype=torch.float32)
        self.sobol_y = torch.tensor(sobol_data.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)
        self.mc_x = torch.tensor(mc_data.iloc[:, :-1].values, dtype=torch.float32)
        self.mc_y = torch.tensor(mc_data.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)

    def get_training_and_test_data(self, sampling_method: SamplingMethod, training_set_size: int) -> TrainingData:
        if sampling_method == SamplingMethod.SOBOL:
            return TrainingData(self.sobol_x[:training_set_size], self.sobol_y[:training_set_size], self.mc_x, self.mc_y)
        elif sampling_method == SamplingMethod.MC:
            return TrainingData(self.mc_x[:training_set_size], self.mc_y[:training_set_size], self.sobol_x, self.sobol_y)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
