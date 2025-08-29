# notebooks/experiment.py
import os
import sys
import time
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import argparse
import json
import glob

sys.path.append(os.path.abspath("src"))
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.enums import OptimizationMethod
from data_classes.experiment import SamplingMethod
from data_classes.scenario import Scenario, ScenarioSettings
from data_classes.training_data import InputData
from data_classes.training_config import TrainingSettings, AdamTrainingConfig
from models import SequentialNeuralNetwork

scenario_map = {
    "projectile": Scenario.PROJECTILE,
    "sumsin6d": Scenario.SUM_SINES_6D,
    "sumsind8d": Scenario.SUM_SINES_8D,
    "sumsin10d": Scenario.SUM_SINES_10D}

sampling_method_map = {
    "halton": SamplingMethod.HALTON,
    "sobol": SamplingMethod.SOBOL,
    "mc": SamplingMethod.MC}

parser = argparse.ArgumentParser(description="Lion tuning experiment")
parser.add_argument("--scenario", type=str, choices=scenario_map.keys(), default="PROJECTILE", help="Scenario to run")
parser.add_argument("--sampling", type=str, choices=sampling_method_map.keys(), default="SOBOL", help="Sampling method")
args, unknown = parser.parse_known_args()


# ---- Experiment context ----
if len(sys.argv) > 1:
    SCENARIO = scenario_map[args.scenario]
    SAMPLING_METHOD = sampling_method_map[args.sampling]
else:
    SCENARIO = Scenario.PROJECTILE
    SAMPLING_METHOD = SamplingMethod.SOBOL

scenario_settings = ScenarioSettings(SCENARIO)
scenario_settings.DATA_PATH = f"data/{SCENARIO.value}"
DATA_PATH = scenario_settings.DATA_PATH
max_epochs = 1000
eps = 1e-8
training_set_size = 4096
trials = 100

# ---- Output paths ----
output_dir = os.path.abspath(os.path.join("data", SCENARIO.value, "output", SAMPLING_METHOD.value, "adam"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "results.csv")
storage = f"sqlite:///{os.path.join(output_dir, 'optuna_study.db')}"
study_name = f"{SCENARIO.value}_{SAMPLING_METHOD.value}_adam"
log_dir_path = os.path.join(output_dir, "logs")

# ---- Results CSV bootstrap (keeps your schema/versioning) ----
def ensure_results_csv():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        version = (df["version"].max() + 1) if not df.empty else 1
        results = df.to_dict(orient="records")
    else:
        df = pd.DataFrame(columns=[
            "version","optimizer","learning_rate","weight_decay","beta_1","beta_2","eps",
            "num_epochs","batch_size","training_set_size","train_error","test_error","train_time",
            "n_layers","layer_depth","activation_function","sampling_method","scenario",
        ])
        version, results = 1, []
    return df, results, version

# ---- Data factory (load inside objective to avoid big pickles) ----
def make_data(training_set_size: int):
    input_data = InputData(DATA_PATH)
    return input_data.get_training_and_test_data(
        sampling_method=SAMPLING_METHOD,
        training_set_size=training_set_size,
    )

# ---- Objective: sample architecture + Adam hyperparams + data sizes ----
def objective(trial: optuna.Trial):
    # Architecture
    n_layers = trial.suggest_int("n_layers", 2, 20)                   # NUM_HIDDEN_LAYERS
    layer_depth = trial.suggest_int("layer_depth", 2,12)
    activation = trial.suggest_categorical("activation", ["Sigmoid", "Tanh"])

    # Data sizes
    # training_set_size = 8192
    # training_set_size = trial.suggest_categorical("training_set_size", [128, 512, 1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    # Adam hyperparams
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    beta1 = trial.suggest_float("beta1", 0.85, 0.99, step=0.01)
    beta2 = trial.suggest_float("beta2", 0.95, 0.999, step=0.001)
    # eps = 1e-8

    # Training length (you can also make this tunable)
    report_epochs_cycle = 25  # for pruning cadence

    arch = NeuralNetworkArchitecture(
        INPUT_DIM=scenario_settings.INPUT_DIM,
        OUTPUT_DIM=scenario_settings.OUTPUT_DIM,
        NUM_HIDDEN_LAYERS=n_layers,
        DEPTH=layer_depth,
        ACTIVATION_FUNCTION=activation,
    )

    cfg = AdamTrainingConfig(
        OPTIMIZER=OptimizationMethod.ADAM,
        LEARNING_RATE=lr,
        REG_PARAM=weight_decay,
        BETAS=(beta1, beta2),
        EPS=eps,
        NUM_EPOCHS=max_epochs,
        BATCH_SIZE=batch_size,
    )

    data = make_data(training_set_size)
    model = SequentialNeuralNetwork(net_arch=arch)

    try:
        model.train(settings=cfg, data=data, trial=trial, report_every=report_epochs_cycle, log_dir=log_dir_path)
    except optuna.TrialPruned:
        raise

    # Optimize test/generalization error
    test_err = model.generalization_error

    # Store useful metadata on the trial
    trial.set_user_attr("train_error", model.train_error)
    trial.set_user_attr("train_time", model.training_results["train_time"])
    trial.set_user_attr("activation_fn", arch.ACTIVATION_FUNCTION)

    return test_err

def get_latest_hyperparams_json(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "best_hyperparameters*.json")))
    if files:
        latest_file = files[-1]
        with open(latest_file, "r") as f:
            return json.load(f), latest_file
    return None, None

if __name__ == "__main__":
    df, results_list, version = ensure_results_csv()

    # Device-aware parallelism: avoid GPU/MPS contention
    use_accel = torch.cuda.is_available() or torch.backends.mps.is_available()
    n_jobs = 1 if use_accel else max(1, (os.cpu_count() or 1) - 0)

    # Reproducible TPE + median pruning
    sampler = TPESampler(seed=42, n_startup_trials=20)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=2, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Load latest hyperparameters if available and enqueue as first trial
    loaded_hyperparams, loaded_file = get_latest_hyperparams_json(output_dir)
    if loaded_hyperparams:
        print(f"Loaded best hyperparameters from: {loaded_file}")
        print("Enqueuing these hyperparameters for first Optuna trial:")
        for k, v in loaded_hyperparams["params"].items():
            print(f"{k}: {v}")
        study.enqueue_trial(loaded_hyperparams["params"])

    # Configure trials
    N_TRIALS = int(os.environ.get("N_TRIALS", f"{trials}"))

    print(f"[Optuna] Starting study '{study_name}' ({N_TRIALS} trials, n_jobs={n_jobs})")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
    print(f"[Optuna] Done in {time.time()-t0:.1f}s. Best value={study.best_value:.6g}")

    # Export all trials (nice for analysis)
    trials_df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs"))
    trials_csv = os.path.join(output_dir, f"optuna_trials_{int(time.time())}.csv")
    trials_df.to_csv(trials_csv, index=False)

    # Append the current best to your results.csv with your original schema
    best = study.best_trial
    p, ua = best.params, best.user_attrs
    results_list.append({
        "version": version,
        "optimizer": "adam",
        "learning_rate": p["learning_rate"],
        "weight_decay": p["weight_decay"],
        "beta_1": p["beta1"],
        "beta_2": p["beta2"],
        "eps": eps,
        "num_epochs": max_epochs,
        "batch_size": p["batch_size"],
        "training_set_size": training_set_size,
        "train_error": ua.get("train_error", None),
        "test_error": best.value,
        "train_time": ua.get("train_time", None),
        "n_layers": p["n_layers"],
        "layer_depth": p["layer_depth"],
        "activation_function": ua.get("activation_fn", "unknown"),
        "sampling_method": SAMPLING_METHOD.value,
        "scenario": SCENARIO.value,
    })

    df_out = pd.DataFrame(results_list).sort_values("test_error").reset_index(drop=True)
    df_out.to_csv(csv_path, index=False)

    print(f"[Optuna] Trials CSV: {trials_csv}")
    print(f"[Optuna] Results CSV updated: {csv_path}")

    print("\nBest hyperparameters found:")
    for k, v in best.params.items():
        print(f"{k}: {v}")
    print("User attributes:")
    for k, v in best.user_attrs.items():
        print(f"{k}: {v}")

    best_hyperparams = {
        "params": best.params,
        "user_attrs": best.user_attrs,
        "test_error": best.value,
    }
    
    base_json_path = os.path.join(output_dir, "best_hyperparameters.json")
    if not os.path.exists(base_json_path):
        best_json_path = base_json_path
    else:
        i = 1
        while True:
            candidate = os.path.join(output_dir, f"best_hyperparameters{i}.json")
            if not os.path.exists(candidate):
                best_json_path = candidate
                break
            i += 1
            
    with open(best_json_path, "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    print(f"Best hyperparameters saved to: {best_json_path}")