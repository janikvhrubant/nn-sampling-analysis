# Neural Network Sampling Analysis

This project explores the impact of different sampling strategies on neural network training, focusing on a comparison between traditional Monte Carlo (MC) sampling and low-discrepancy Quasi-Monte Carlo (QMC) sequences.

The study spans five diverse scenarios designed to test generalization across varying problem types and dimensionalities:
- **Six-dimensional sum of sines** – a smooth, high-dimensional function.
- **BSPDE** – a backward stochastic partial differential equation.
- **Airfoil pressure prediction** – a physics-informed, real-world inspired scenario.
- **Projectile motion** – a classical mechanics problem with known analytical solution.

By analyzing training convergence, generalization error, and robustness across these problems, this repository provides insight into the role of sampling in neural network performance.

## Goals
- Evaluate how QMC sampling compares to standard MC for neural network training.
- Analyze performance across low- and high-dimensional, smooth and complex problem types.
- Understand whether better point distributions can lead to better generalization with fewer training points.

## Contents
- 📁 `data/` – Sampling point generation scripts (MC and QMC)
- 📁 `models/` – Neural network architectures and training loops
- 📁 `scenarios/` – Problem definitions and ground truth evaluations
- 📊 `results/` – Evaluation metrics and visualizations
