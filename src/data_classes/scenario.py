from dataclasses import dataclass
import os
import numpy as np
from .enums import Scenario

@dataclass
class ScenarioSettings:
    INPUT_DIM: int
    OUTPUT_DIM: int
    SCENARIO: Scenario
    DATA_PATH: str
    TRINING_SET_SIZES: list

    def __init__(self, scenario: Scenario):
        self.SCENARIO = scenario
        self.DATA_PATH = os.path.abspath(os.path.join('..', 'data', scenario.value))

        match scenario:
            case Scenario.BSPDE:
                self.INPUT_DIM = 5
                self.OUTPUT_DIM = 1
                self.TRINING_SET_SIZES = (np.logspace(5, 11, 7, base=2).astype(int)).tolist()
            case Scenario.SUM_SINES:
                self.INPUT_DIM = 6
                self.OUTPUT_DIM = 1
                self.TRINING_SET_SIZES = (np.logspace(7, 13, 7, base=2).astype(int)).tolist()
            case Scenario.PROJECTILE:
                self.INPUT_DIM = 3
                self.OUTPUT_DIM = 1
                self.TRINING_SET_SIZES = (np.logspace(4, 10, 7, base=2).astype(int)).tolist()
            case Scenario.AIRFOIL:
                self.INPUT_DIM = 4
                self.OUTPUT_DIM = 1
                self.TRINING_SET_SIZES = (np.logspace(2, 8, 7, base=2).astype(int)).tolist()
            case _:
                raise ValueError(f"Unknown scenario: {scenario}")
