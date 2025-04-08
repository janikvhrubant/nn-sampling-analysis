# Neural Network Sampling Analysis

This project explores the impact of different sampling strategies on neural network training, focusing on a comparison between traditional Monte Carlo (MC) sampling and low-discrepancy Quasi-Monte Carlo (SOBOL) sequences.

The study spans five diverse scenarios designed to test generalization across varying problem types and dimensionalities:
- **Six-dimensional sum of sines** – a smooth, high-dimensional function.
- **BSPDE** – a backward stochastic partial differential equation. (tbd)
- **Airfoil pressure prediction** – a physics-informed, real-world inspired scenario. (tbd)
- **Projectile motion** – a classical mechanics problem with known analytical solution. (tbd)

By analyzing training convergence, generalization error, and robustness across these problems, this repository provides insight into the role of sampling in neural network performance.

## Goals
- Compare SOBOL and MC sampling techniques in neural network training.
- Evaluate performance on a high-dimensional smooth function.
- Analyze whether improved sampling leads to better generalization with fewer data points.

## Contents
- 📁 `data/` – Contains input sampling data (MC and SOBOL) and output results from training experiments.
- 📁 `notebooks/` – Jupyter notebooks for sampling, running experiments, and evaluating results.
- 📁 `src/` – Core source code including model definitions, sampling logic, and structured data/config classes.
  - 📁 `data_classes/` – Data models and configuration structures for experiments, scenarios, and training.
  - 📄 `models.py` – Neural network architecture definitions.
  - 📄 `sampling.py` – Logic for generating MC and SOBOL samples.

## Project Structure

```text
.
├── README.md                      # Project documentation
├── requirements.txt               # Contains all required Python packages
├── data/
│   └── sum_sines/
│       ├── input/                 # Input samples (MC & SOBOL)
│       │   ├── mc_sample.csv
│       │   └── sobol_sample.csv
│       └── output/                # Evaluation results for various training durations
│           ├── mc_1000epochs_results.csv
│           ├── mc_100epochs_results.csv
│           ├── sobol_1000epochs_results.csv
│           ├── sobol_100epochs_results.csv
│           └── together.csv
├── notebooks/                     # Jupyter notebooks for experimentation and evaluation
│   ├── eval.ipynb
│   ├── evaluate.ipynb
│   ├── experiment.ipynb
│   └── sampling.ipynb
└── src/                           # Source code
    ├── data_classes/              # Custom data classes and config definitions
    │   ├── __init__.py
    │   ├── architecture.py
    │   ├── enums.py
    │   ├── experiment.py
    │   ├── scenario.py
    │   ├── training_config.py
    │   ├── training_data.py
    │   └── training_results.py
    ├── models.py                  # Model architectures
    └── sampling.py                # Sampling logic for MC and Sobol sequences
```

## How to Use
1. **Setup**: Run `pip install -r requirements.txt` with Python 3.10 or higher to install required dependencies.
2. **Sampling**: Generate input data using MC or SOBOL in `notebooks/sampling.ipynb`.
3. **Training**: Run experiments and train models via `notebooks/experiment.ipynb`.
4. **Evaluation**: Analyze model performance and compare results in `notebooks/evaluate.ipynb`.

## Future Work
- [ ] Extend the evaluation to more problem types (e.g., PDEs, physics-based models).
- [ ] Implement more QMC-strategies such as Lattice-Sequence
- [ ] Maybe: Introduce adaptive sampling during training
