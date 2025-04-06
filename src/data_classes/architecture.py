from dataclasses import dataclass
from torch import nn

@dataclass
class NeuralNetworkArchitecture:
    INPUT_DIM: int
    OUTPUT_DIM: int
    NUM_HIDDEN_LAYERS: int
    DEPTH: int
    ACTIVATION_FUNCTION: nn.Module
    BATCH_NORMALIZATION: bool = True
