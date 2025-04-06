# from dataclasses import dataclass
# from torch import nn, Tensor
# from enum import Enum
# import os
# import numpy as np
# import pandas as pd
# import torch
# from typing import Tuple


# class Scenario(Enum):
#     BSPDE = "bspde"
#     SUM_SINES = "sum_sines"
#     PROJECTILE = "projectile"
#     AIRFOIL = "airfoil"


# @dataclass
# class ScenarioSettings:
#     INPUT_DIM: int
#     OUTPUT_DIM: int
#     SCENARIO: Scenario
#     DATA_PATH: str
#     TRINING_SET_SIZES: list

#     def __init__(self, scenario: Scenario):
#         if scenario == Scenario.BSPDE:
#             self.INPUT_DIM = 5
#             self.OUTPUT_DIM = 1
#             self.DATA_PATH = os.path.abspath(os.path.join('..', 'data', scenario.value))
#             self.TRINING_SET_SIZES = (np.logspace(5, 11, 7, base=2).astype(int)).tolist()
#         elif scenario == Scenario.SUM_SINES:
#             self.INPUT_DIM = 6
#             self.OUTPUT_DIM = 1
#             self.DATA_PATH = os.path.abspath(os.path.join('..', 'data', scenario.value))
#             self.TRINING_SET_SIZES = (np.logspace(7, 13, 7, base=2).astype(int)).tolist()
#         elif scenario == Scenario.PROJECTILE:
#             self.INPUT_DIM = 3
#             self.OUTPUT_DIM = 1
#             self.DATA_PATH = os.path.abspath(os.path.join('..', 'data', scenario.value))
#             self.TRINING_SET_SIZES = (np.logspace(4, 10, 7, base=2).astype(int)).tolist()
#         elif scenario == Scenario.AIRFOIL:
#             self.INPUT_DIM = 4
#             self.OUTPUT_DIM = 1
#             self.DATA_PATH = os.path.abspath(os.path.join('..', 'data', scenario.value))
#             self.TRINING_SET_SIZES = (np.logspace(2, 8, 7, base=2).astype(int)).tolist()
#         else:
#             raise ValueError(f"Unknown scenario: {scenario}")
#         self.SCENARIO = scenario


# class SamplingMethod(Enum):
#     QMC = "qmc"
#     MC = "mc"


# @dataclass
# class Experiment:
#     SAMPLING_METHOD: SamplingMethod
#     SCENARIO: Scenario


# @dataclass
# class NeuralNetworkArchitecture:
#     INPUT_DIM: int
#     OUTPUT_DIM: int
#     NUM_HIDDEN_LAYERS: int
#     DEPTH: int
#     ACTIVATION_FUNCTION: nn.Module
#     BATCH_NORMALIZATION: bool = True


# class OptimizationMethod(Enum):
#     SGD = "sgd"
#     NAG = "nag"
#     ADAM = "adam"
#     RMSPROP = "rmsprop"


# @dataclass
# class BaseTrainingConfig:
#     OPTIMIZER: OptimizationMethod
#     LEARNING_RATE: float
#     REG_PARAM: float
#     NUM_EPOCHS: int


# @dataclass
# class SGDTrainingConfig(BaseTrainingConfig):
#     MOMENTUM: float = 0.0
#     NESTEROV: bool = False


# @dataclass
# class AdamTrainingConfig(BaseTrainingConfig):
#     BETAS: Tuple[float, float] = (0.9, 0.99)
#     EPS: float = 1e-8


# @dataclass
# class RMSpropTrainingConfig(BaseTrainingConfig):
#     MOMENTUM: float = 0.0
#     ALPHA: float = 0.99
#     EPS: float = 1e-8
#     CENTERED: bool = False





# @dataclass
# class TrainingData:
#     train_x: Tensor
#     train_y: Tensor
#     test_x: Tensor
#     test_y: Tensor


# @dataclass
# class InputData:
#     qmc_x: Tensor
#     qmc_y: Tensor
#     mc_x: Tensor
#     mc_y: Tensor

#     def __init__(self, data_path: str):
#         qmc_path = os.path.join(data_path, 'input', 'qmc_train.csv')
#         mc_path = os.path.join(data_path, 'input', 'mc_train.csv')

#         qmc_data = pd.read_csv(qmc_path)
#         mc_data = pd.read_csv(mc_path)

#         self.qmc_x = torch.tensor(qmc_data.iloc[:, :-1].values, dtype=torch.float32)
#         self.qmc_y = torch.tensor(qmc_data.iloc[:, -1].values, dtype=torch.float32)
#         self.mc_x = torch.tensor(mc_data.iloc[:, :-1].values, dtype=torch.float32)
#         self.mc_y = torch.tensor(mc_data.iloc[:, -1].values, dtype=torch.float32)

#     def get_training_and_test_data(self, sampling_method: SamplingMethod, training_set_size: int) -> TrainingData:
#         if sampling_method == SamplingMethod.QMC:
#             return TrainingData(
#                 train_x=self.qmc_x[:training_set_size],
#                 train_y=self.qmc_y[:training_set_size],
#                 test_x=self.mc_x,
#                 test_y=self.mc_y
#             )
#         elif sampling_method == SamplingMethod.MC:
#             return TrainingData(
#                 train_x=self.mc_x[:training_set_size],
#                 train_y=self.mc_y[:training_set_size],
#                 test_x=self.qmc_x,
#                 test_y=self.qmc_y
#             )
#         else:
#             raise ValueError(f"Unknown sampling method: {sampling_method}")
