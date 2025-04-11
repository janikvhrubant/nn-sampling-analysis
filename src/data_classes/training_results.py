from dataclasses import dataclass
from .training_config import AdamTrainingConfig

@dataclass
class TrainingResult:
    adam_config: AdamTrainingConfig
    train_error: float
    test_error: float
    num_layers: int
    training_set_size: int
