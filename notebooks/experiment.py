# notebooks/experiment.py
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.append(os.path.abspath(os.path.join('src')))
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.enums import OptimizationMethod
from data_classes.experiment import SamplingMethod
from data_classes.scenario import Scenario, ScenarioSettings
from data_classes.training_data import InputData
from data_classes.training_config import AdamTrainingConfig, TrainingSettings, LionTrainingConfig
from models import SequentialNeuralNetwork

# ---- Static experiment config (pass only primitives/enums to workers) ----
SCENARIO = Scenario.PROJECTILE
SAMPLING_METHOD = SamplingMethod.SOBOL

scenario_settings = ScenarioSettings(SCENARIO)
scenario_settings.DATA_PATH = f'data/{SCENARIO.value}'

DATA_PATH = scenario_settings.DATA_PATH  # pass to workers

# NN Architectures
widths = [4, 6, 8]
depths = [2,4,6,8,10]
activation_functions = [torch.nn.Sigmoid, torch.nn.Tanh]

# Optimizer settings
# Adam
adam_beta1s = [0.9]
adam_beta2s = [0.999]
adam_lr_wd_combos = [(1e-3, 0), (1e-3, 1e-6), (1e-3, 1e-5), (3e-4, 0), (3e-4, 1e-6), (3e-4, 1e-5)]
adam_epsilons = [1e-8]

# Lion
lion_beta1s = [0.9]
lion_beta2s = [0.99]
lion_lr_wd_combos = [(3e-3, 0), (3e-3, 1e-6), (1e-3, 0), (1e-3, 1e-6), (3e-3, 1e-5), (1e-3, 1e-5)]

# Learning settings:
training_set_sizes = [128, 512, 1024, 2048, 4096, 8192]
batch_sizes = [64, 256, 1024]
max_epochs = 1500

all_training_settings = []
for width, depth, activation_func, train_set_size, batch_size in product(
    widths, depths, activation_functions, training_set_sizes, batch_sizes
):
    nn_arch = NeuralNetworkArchitecture(
        INPUT_DIM=scenario_settings.INPUT_DIM,
        OUTPUT_DIM=scenario_settings.OUTPUT_DIM,
        NUM_HIDDEN_LAYERS=width,
        DEPTH=depth,
        ACTIVATION_FUNCTION=activation_func,
    )

    # Adam configs
    for beta1, beta2, lr_wd, epsilon in product(adam_beta1s, adam_beta2s, adam_lr_wd_combos, adam_epsilons):
        learning_rate, weight_decay = lr_wd

        training_config = AdamTrainingConfig(
            OPTIMIZER=OptimizationMethod.ADAM,
            LEARNING_RATE=learning_rate,
            REG_PARAM=weight_decay,
            BETAS=(beta1, beta2),
            EPS=epsilon,
            NUM_EPOCHS=max_epochs,
            BATCH_SIZE=batch_size,
        )

        all_training_settings.append(
            TrainingSettings(nn_architecture=nn_arch, training_config=training_config, training_set_size=train_set_size)
        )

    # Lion configs
    for beta1, beta2, lr_wd in product(lion_beta1s, lion_beta2s, lion_lr_wd_combos):
        learning_rate, weight_decay = lr_wd

        training_config = LionTrainingConfig(
            OPTIMIZER=OptimizationMethod.LION,
            LEARNING_RATE=learning_rate,
            REG_PARAM=weight_decay,
            BETAS=(beta1, beta2),
            NUM_EPOCHS=max_epochs,
            BATCH_SIZE=batch_size,
        )

        all_training_settings.append(
            TrainingSettings(nn_architecture=nn_arch, training_config=training_config, training_set_size=train_set_size)
        )


def train_single(ts: TrainingSettings, data_path: str, sampling_method: SamplingMethod, scenario: Scenario, version: int):
    """Train a single NN configuration and return the results dict.
    NOTE: Re-create InputData INSIDE the worker to avoid pickling large tensors across processes.
    """
    # Load data lazily in each process (small object passed from parent)
    input_data = InputData(data_path)
    nn = SequentialNeuralNetwork(net_arch=ts.nn_architecture)

    training_data = input_data.get_training_and_test_data(
        sampling_method=sampling_method,
        training_set_size=ts.training_set_size,
    )

    nn.train(settings=ts.training_config, data=training_data)
    nn.training_results["version"] = version
    nn.training_results["scenario"] = scenario.value
    nn.training_results["sampling_method"] = sampling_method.value
    return nn.training_results


if __name__ == "__main__":
    # Output directory setup
    output_dir = os.path.abspath(os.path.join("data", SCENARIO.value, "output"))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        version = (df["version"].max() + 1) if not df.empty else 1
        results_list = df.to_dict(orient="records")
    else:
        df = pd.DataFrame(
            columns=[
                "version",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "beta_1",
                "beta_2",
                "eps",
                "num_epochs",
                "batch_size",
                "training_set_size",
                "train_error",
                "test_error",
                "train_time",
                "n_layers",
                "layer_depth",
                "activation_function",
                "sampling_method",
                "scenario",
            ]
        )
        version = 1
        results_list = []

    # Decide parallelism: if CUDA present, avoid multi-proc GPU contention
    use_cuda = torch.cuda.is_available()
    max_workers = 1 if use_cuda else os.cpu_count() or 1

    # Warm-up run (safe when max_workers == 1 on CUDA)
    _ = train_single(all_training_settings[0], DATA_PATH, SAMPLING_METHOD, SCENARIO, version)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        func = partial(
            train_single,
            data_path=DATA_PATH,
            sampling_method=SAMPLING_METHOD,
            scenario=SCENARIO,
            version=version,
        )
        for result in tqdm(
            executor.map(func, all_training_settings),
            total=len(all_training_settings),
            desc="Training Progress",
        ):
            results.append(result)

    results_list.extend(results)
    df_results = pd.DataFrame(results_list)
    df_results.sort_values("test_error", inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    df_results.to_csv(csv_path, index=False)
