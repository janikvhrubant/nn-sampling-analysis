from enum import Enum

class Scenario(Enum):
    BSPDE = "bspde"
    SUM_SINES_6D = "sum_sines_6d"
    SUM_SINES_8D = "sum_sines_8d"
    SUM_SINES_10D = "sum_sines_10d"
    PROJECTILE = "projectile"
    AIRFOIL = "airfoil"

class SamplingMethod(Enum):
    HALTON = "halton"
    SOBOL = "sobol"
    MC = "mc"

class OptimizationMethod(Enum):
    LION = "lion"
    SGD = "sgd"
    NAG = "nag"
    ADAM = "adam"
    RMSPROP = "rmsprop"
