from dataclasses import dataclass
from .enums import Scenario, SamplingMethod

@dataclass
class Experiment:
    SAMPLING_METHOD: SamplingMethod
    SCENARIO: Scenario
