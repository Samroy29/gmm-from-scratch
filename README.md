# gmm-from-scratch
This repository contains a from-scratch implementation of a Gaussian Mixture Model (GMM) in Python. It is designed to demonstrate the EM algorithm, handling of diagonal covariance matrices, and numerical stability techniques for robust computation. The code is accelerated with Numba for parallelization and efficiency.

Features
Full EM algorithm implementation: E-step (assignment probabilities) and M-step (parameter maximization).
Supports diagonal covariance matrices for computational efficiency.
Includes stabilization fixes to prevent collapsing variances and zero probabilities.
Tracks log-likelihood evolution over iterations.
Modular functions for probability density, log-likelihood, parameter updates, and cluster assignment.
Easy to experiment with different numbers of clusters and temperature schedules for soft assignments.
Purpose

This project is intended for learning and demonstration of GMMs, EM optimization, and numerical techniques in probabilistic models. It focuses on the core ML algorithm, with all domain-specific or business logic removed.

![GMM Cluster Plot](images/Figure_1.png)
