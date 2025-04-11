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
    TRAINING_SET_SIZES: list
    COL_NAMES: list[str]

    def __init__(self, scenario: Scenario):
        self.SCENARIO = scenario
        self.DATA_PATH = os.path.abspath(os.path.join('..', 'data', scenario.value))

        match scenario:
            case Scenario.BSPDE:
                self.INPUT_DIM = 5
                self.OUTPUT_DIM = 1
                self.TRAINING_SET_SIZES = (np.logspace(5, 11, 7, base=2).astype(int)).tolist()
            case Scenario.SUM_SINES:
                self.INPUT_DIM = 6
                self.OUTPUT_DIM = 1
                self.TRAINING_SET_SIZES = (np.logspace(7, 13, 7, base=2).astype(int)).tolist()
                self.COL_NAMES = ['dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5', 'dim_6', 'sum_sines']
            case Scenario.PROJECTILE:
                self.INPUT_DIM = 7
                self.OUTPUT_DIM = 1
                self.TRAINING_SET_SIZES = (np.logspace(4, 10, 7, base=2).astype(int)).tolist()
                self.COL_NAMES = ['density', 'radius', 'drag_coeff', 'mass', 'height', 'alpha', 'velocity', 'distance']
            case Scenario.AIRFOIL:
                self.INPUT_DIM = 4
                self.OUTPUT_DIM = 1
                self.TRAINING_SET_SIZES = (np.logspace(2, 8, 7, base=2).astype(int)).tolist()
            case _:
                raise ValueError(f"Unknown scenario: {scenario}")
