from dataclasses import dataclass
from typing import Tuple
from .enums import OptimizationMethod
from .architecture import NeuralNetworkArchitecture

# @dataclass
# class BaseTrainingConfig:
#     OPTIMIZER: OptimizationMethod
#     LEARNING_RATE: float
#     REG_PARAM: float
#     NUM_EPOCHS: int

@dataclass
class BaseTrainingConfig:
    OPTIMIZER: OptimizationMethod
    LEARNING_RATE: float
    REG_PARAM: float
    NUM_EPOCHS: int
    BATCH_SIZE: int = None

@dataclass
class TrainingSettings:
    nn_architecture: NeuralNetworkArchitecture
    training_config: BaseTrainingConfig
    training_set_size: int

@dataclass
class SGDTrainingConfig(BaseTrainingConfig):
    MOMENTUM: float = 0.0
    NESTEROV: bool = False

@dataclass
class AdamTrainingConfig(BaseTrainingConfig):
    BETAS: Tuple[float, float] = (0.9, 0.99)
    EPS: float = 1e-8

@dataclass
class LionTrainingConfig(BaseTrainingConfig):
    BETAS: Tuple[float, float] = (0.9, 0.99)

@dataclass
class RMSpropTrainingConfig(BaseTrainingConfig):
    MOMENTUM: float = 0.0
    ALPHA: float = 0.99
    EPS: float = 1e-8
    CENTERED: bool = False
