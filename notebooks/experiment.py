import os
import sys
import pandas as pd
import torch
from datetime import datetime
import tqdm
from itertools import product

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.enums import OptimizationMethod
from data_classes.experiment import Experiment, SamplingMethod
from data_classes.scenario import Scenario, ScenarioSettings
from data_classes.training_data import InputData
from data_classes.training_config import AdamTrainingConfig, TrainingSettings, LionTrainingConfig
from models import SequentialNeuralNetwork

experiment = Experiment(
    SAMPLING_METHOD=SamplingMethod.MC,
    SCENARIO=Scenario.SUM_SINES_6D
)

scenario_settings = ScenarioSettings(experiment.SCENARIO)

input_data = InputData(scenario_settings.DATA_PATH)

# NN Architectures
widths = [2,8,16,64]
depths = [2,4,8]
activation_functions = [torch.nn.Sigmoid, torch.nn.Tanh]

# Optimizer settings
# Adam
adam_beta1s = [0.9]
adam_beta2s = [0.999]
adam_lr_wd_combos = [(1e-3,0), (1e-3,1e-6), (1e-3,1e-5), (3e-4,0), (3e-4,1e-6), (3e-4,1e-5)]
adam_epsilons = [1e-8]

# Lion
lion_beta1s = [0.9]
lion_beta2s = [0.99]
lion_lr_wd_combos = [(3e-3,0),(3e-3, 1e-6),(1e-3,0),(1e-3,1e-6),(3e-3,1e-5),(1e-3,1e-5)]


# Learing settings:
training_set_sizes = [128,512,2048,8192]
batch_sizes = [64,256,512,1024,2048]
epochs = [50,100,250,500,750,1000,1250,1500]

all_training_settings = []
for width, depth, activation_func, train_set_size, batch_size, epoch in product(widths, depths, activation_functions, training_set_sizes, batch_sizes, epochs):
    
    nn_arch = NeuralNetworkArchitecture(
        INPUT_DIM=scenario_settings.INPUT_DIM,
        OUTPUT_DIM=scenario_settings.OUTPUT_DIM,
        NUM_HIDDEN_LAYERS=width,
        DEPTH=depth,
        ACTIVATION_FUNCTION=activation_func
    )

    for beta1, beta2, lr_wd, epsilon in product(adam_beta1s, adam_beta2s, adam_lr_wd_combos, adam_epsilons):
        learning_rate, weight_decay = lr_wd
        
        training_config = AdamTrainingConfig(
            OPTIMIZER=OptimizationMethod.ADAM,
            LEARNING_RATE=learning_rate,
            REG_PARAM=weight_decay,
            BETAS=(beta1, beta2),
            EPS=epsilon,
            NUM_EPOCHS=epoch,
            BATCH_SIZE=batch_size
        )

        all_training_settings.append(TrainingSettings(
            nn_architecture=nn_arch,
            training_config=training_config,
            training_set_size=train_set_size
        ))
    
    for beta1, beta2, lr_wd in product(lion_beta1s, lion_beta2s, lion_lr_wd_combos):

        training_config = LionTrainingConfig(
            OPTIMIZER=OptimizationMethod.LION,
            LEARNING_RATE=learning_rate,
            REG_PARAM=weight_decay,
            BETAS=(beta1, beta2),
            NUM_EPOCHS=epoch,
            BATCH_SIZE=batch_size
        )

        all_training_settings.append(TrainingSettings(
            nn_architecture=nn_arch,
            training_config=training_config,
            training_set_size=train_set_size
        ))

output_dir = os.path.abspath(os.path.join('..', 'data', experiment.SCENARIO.value, 'output'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'results.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    version = df['version'].max() + 1
    results_list = df.to_dict(orient='records')
else:
    df = pd.DataFrame(columns=['version','optimizer', 'learning_rate', 'weight_decay', 'beta_1', 'beta_2', 'eps', 'num_epochs', 'batch_size', 'training_set_size', 'train_error', 'test_error', 'train_time', 'n_layers', 'layer_depth', 'activation_function', 'sampling_method', 'scenario'])
    version = 1
    results_list = []

counter = 0

for ts in tqdm.tqdm(all_training_settings, desc="Training Progress"):
    nn = SequentialNeuralNetwork(
        net_arch=ts.nn_architecture
    )
    training_data = input_data.get_training_and_test_data(
        sampling_method=experiment.SAMPLING_METHOD,
        training_set_size=ts.training_set_size
    )
    nn.train(settings=ts.training_config, data=training_data)
    nn.training_results['version'] = version
    nn.training_results['scenario'] = experiment.SCENARIO.value
    results_list.append(nn.training_results)
    df_results = pd.DataFrame(results_list)

    df_results.sort_values('test_error', inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    df_results.to_csv(csv_path, index=False)