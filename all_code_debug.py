import os
import sys
import pandas as pd
import torch
from itertools import product

sys.path.append(os.path.abspath(os.path.join('src')))
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.enums import OptimizationMethod
from data_classes.experiment import Experiment, SamplingMethod
from data_classes.scenario import Scenario, ScenarioSettings
from data_classes.training_data import InputData
from data_classes.training_config import AdamTrainingConfig, TrainingSettings
from models import SequentialNeuralNetwork
from tqdm import tqdm

# experiment = Experiment(
#     SAMPLING_METHOD=SamplingMethod.SOBOL,
#     SCENARIO=Scenario.PROJECTILE
# )

experiment = Experiment(
    SAMPLING_METHOD=SamplingMethod.HALTON,
    SCENARIO=Scenario.PROJECTILE
)

scenario = Scenario.SUM_SINES_6D
experiments = [
    Experiment(
        SAMPLING_METHOD=SamplingMethod.SOBOL,
        SCENARIO=scenario
    ),
    Experiment(
        SAMPLING_METHOD=SamplingMethod.HALTON,
        SCENARIO=scenario
    ),
    Experiment(
        SAMPLING_METHOD=SamplingMethod.MC,
        SCENARIO=scenario
    )
]

scenario_settings = ScenarioSettings(experiment.SCENARIO)

input_data = InputData(scenario_settings.DATA_PATH)

# widths = [24]
# depths = [16]
# learning_rates = [0.001]
# lambdas = [1e-07]
# training_set_sizes = [128]
widths = [6,12,24]
depths = [4,8,16]
learning_rates = [0.01,0.001]
lambdas = [1.0e-04,1.0e-05,1.0e-06,1e-07]
# training_set_sizes = scenario_settings.TRAINING_SET_SIZES
training_set_sizes = [2**i for i in range(5,14)]

all_training_settings = []
training_settings = []

for experiment, width, depth, learning_rate, lambda_, training_set_size in product(experiments, widths, depths, lambdas, learning_rates, training_set_sizes):
    nn_arch = NeuralNetworkArchitecture(
        INPUT_DIM=scenario_settings.INPUT_DIM,
        OUTPUT_DIM=scenario_settings.OUTPUT_DIM,
        NUM_HIDDEN_LAYERS=width,
        DEPTH=depth,
        ACTIVATION_FUNCTION=torch.nn.Sigmoid
    )
    training_config = AdamTrainingConfig(
        OPTIMIZER=OptimizationMethod.ADAM,
        LEARNING_RATE=learning_rate,
        REG_PARAM=lambda_,
        MAX_EPOCHS=1500
    )

    training_settings.append(TrainingSettings(
        nn_architecture=nn_arch,
        training_config=training_config,
        training_set_size=training_set_size
    ))

training_results_list = []

output_dir = os.path.abspath(os.path.join('data', experiment.SCENARIO.value, 'output'))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, f'{experiment.SAMPLING_METHOD.value}_{training_set_size}samples_results.csv')
together_csv_path = os.path.join(output_dir, f'together2.csv')

for ts in tqdm(training_settings, desc="Training progress", unit="config"):
    nn = SequentialNeuralNetwork(
        net_arch=ts.nn_architecture
    )
    training_data = input_data.get_training_and_test_data(
        sampling_method=experiment.SAMPLING_METHOD,
        training_set_size=ts.training_set_size
    )
    nn.train(settings=ts.training_config, data=training_data)
    training_results_list.extend(nn.training_results)
    df_results = pd.DataFrame(training_results_list)

    df_results.sort_values('test_error', inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    df_results.to_csv(csv_path, index=False)

    df_results['SamplingMethod'] = experiment.SAMPLING_METHOD
    df_results['NumSamples'] = ts.training_set_size
    df_results.to_csv(together_csv_path)


