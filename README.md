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

```bash
git clone https://github.com/Samroy29/gmm-from-scratch.git
cd gmm-from-scratch
pip install numpy numba matplotlib

import numpy as np
import matplotlib.pyplot as plt
from gmm_module import Gaussian_Mixture_Model  # replace with your module name

# Generate random sample data
np.random.seed(0)
c1 = np.random.normal([5,5], 0.5, (100,2))
c2 = np.random.normal([0,5], 0.5, (100,2))
c3 = np.random.normal([0,0], 0.5, (100,2))
observations = np.vstack([c1, c2, c3])

# Fit GMM
gmm_result = Gaussian_Mixture_Model.GMM(observations, num_of_clusters=3)

print("Final Cluster Centers:")
print(gmm_result["final_centers"])
print("\nCluster Weights:")
print(gmm_result["weights"])
print("\nLog-Likelihood Evolution:")
print(gmm_result["log_likelihood_evolution"])

# Plot clustering result
assignments = gmm_result["assignments"]
cluster_assignments = np.argmax(assignments, axis=1)
plt.scatter(observations[:,0], observations[:,1], c=cluster_assignments, cmap="viridis", alpha=0.6)
plt.scatter(gmm_result["final_centers"][:,0], gmm_result["final_centers"][:,1], c="red", marker="X", s=100)
plt.title("GMM Cluster Assignment")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("Figure_1.png")
plt.show()
