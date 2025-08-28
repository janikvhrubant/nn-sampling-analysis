# notebooks/experiment.py
import os
import sys
import time
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.append(os.path.abspath("src"))
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.enums import OptimizationMethod
from data_classes.experiment import SamplingMethod
from data_classes.scenario import Scenario, ScenarioSettings
from data_classes.training_data import InputData
from data_classes.training_config import TrainingSettings, LionTrainingConfig
from models import SequentialNeuralNetwork

# ---- Static experiment config ----
SCENARIO = Scenario.PROJECTILE
SAMPLING_METHOD = SamplingMethod.SOBOL

scenario_settings = ScenarioSettings(SCENARIO)
scenario_settings.DATA_PATH = f"data/{SCENARIO.value}"
DATA_PATH = scenario_settings.DATA_PATH

# ---- Output paths ----
output_dir = os.path.abspath(os.path.join("data", SCENARIO.value, "output"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "results.csv")

# ---- Results CSV bootstrap ----
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

# ---- Data factory (load inside objective) ----
def make_data(training_set_size: int):
    input_data = InputData(DATA_PATH)
    return input_data.get_training_and_test_data(
        sampling_method=SAMPLING_METHOD,
        training_set_size=training_set_size,
    )

# ---- Objective for Lion ----
def objective(trial: optuna.Trial):
    # Architecture search space (mirrors your old grids but continuous where useful)
    n_layers = trial.suggest_int("n_layers", 2, 10)  # NUM_HIDDEN_LAYERS
    layer_depth = trial.suggest_categorical("layer_depth", [2, 4, 6, 8, 10])
    activation = trial.suggest_categorical("activation", [torch.nn.Sigmoid, torch.nn.Tanh])

    # Data sizes
    training_set_size = trial.suggest_categorical("training_set_size", [128, 512, 1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 256, 1024])

    # Lion hyperparams (expanded around your presets)
    beta1 = trial.suggest_float("beta1", 0.88, 0.92, step=0.01)   # around 0.9
    beta2 = trial.suggest_float("beta2", 0.98, 0.995, step=0.001) # around 0.99
    lr = trial.suggest_float("lr", 8e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 3e-5, log=True)

    # Epochs and reporting cadence
    max_epochs = 1500
    report_every = 25

    arch = NeuralNetworkArchitecture(
        INPUT_DIM=scenario_settings.INPUT_DIM,
        OUTPUT_DIM=scenario_settings.OUTPUT_DIM,
        NUM_HIDDEN_LAYERS=n_layers,
        DEPTH=layer_depth,
        ACTIVATION_FUNCTION=activation,
    )

    cfg = LionTrainingConfig(
        OPTIMIZER=OptimizationMethod.LION,
        LEARNING_RATE=lr,
        REG_PARAM=weight_decay,
        BETAS=(beta1, beta2),
        NUM_EPOCHS=max_epochs,
        BATCH_SIZE=batch_size,
    )

    data = make_data(training_set_size)
    model = SequentialNeuralNetwork(net_arch=arch)

    try:
        model.train(settings=cfg, data=data, trial=trial, report_every=report_every)
    except optuna.TrialPruned:
        raise

    # Optimize generalization error
    test_err = model.generalization_error

    # Store extras
    trial.set_user_attr("train_error", model.train_error)
    trial.set_user_attr("train_time", model.training_results["train_time"])
    trial.set_user_attr("activation_fn", arch.ACTIVATION_FUNCTION.__name__)

    return test_err

if __name__ == "__main__":
    df, results_list, version = ensure_results_csv()

    # Device-aware parallelism (single process on GPU/MPS)
    use_accel = torch.cuda.is_available() or torch.backends.mps.is_available()
    n_jobs = 1 if use_accel else max(1, (os.cpu_count() or 1))

    # In-memory study (no persistent storage)
    sampler = TPESampler(seed=42, n_startup_trials=20, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    N_TRIALS = int(os.environ.get("N_TRIALS", "80"))

    print(f"[Optuna] Starting (no storage) with {N_TRIALS} trials, n_jobs={n_jobs}")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
    print(f"[Optuna] Done in {time.time()-t0:.1f}s. Best value={study.best_value:.6g}")

    # Export all trials to CSV for analysis
    trials_df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs"))
    trials_csv = os.path.join(output_dir, f"optuna_trials_{int(time.time())}.csv")
    trials_df.to_csv(trials_csv, index=False)

    # Append current best to your results.csv with your original schema
    best = study.best_trial
    p, ua = best.params, best.user_attrs
    results_list.append({
        "version": version,
        "optimizer": "lion",
        "learning_rate": p["lr"],
        "weight_decay": p["weight_decay"],
        "beta_1": p["beta1"],
        "beta_2": p["beta2"],
        "eps": None,
        "num_epochs": 1500,
        "batch_size": p["batch_size"],
        "training_set_size": p["training_set_size"],
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
