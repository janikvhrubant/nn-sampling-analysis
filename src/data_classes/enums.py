from enum import Enum

class Scenario(Enum):
    BSPDE = "bspde"
    SUM_SINES = "sum_sines"
    PROJECTILE = "projectile"
    AIRFOIL = "airfoil"

class SamplingMethod(Enum):
    QMC = "qmc"
    MC = "mc"

class OptimizationMethod(Enum):
    SGD = "sgd"
    NAG = "nag"
    ADAM = "adam"
    RMSPROP = "rmsprop"
