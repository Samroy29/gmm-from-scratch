import numpy as np
import math
from numba import njit, prange
import matplotlib.pyplot as plt


@njit
def probability_density(x, mean, diag_cov):
    diff = x - mean

    # ---- STABILIZATION FIX ----
    # ensure no zero variances inside density computation
    safe_diag = np.empty_like(diag_cov)
    for i in range(len(diag_cov)):
        if diag_cov[i] < 1e-6:
            safe_diag[i] = 1e-6
        else:
            safe_diag[i] = diag_cov[i]
    # --------------------------------

    exponent = -0.5 * np.sum((diff**2) / safe_diag)
    
    # prod durch Schleife ersetzen, damit Numba kein Problem hat
    prod_diag = np.float64(1.0)
    for i in range(len(safe_diag)):
        prod_diag *= safe_diag[i]
        
    normalization = 1.0 / math.sqrt((2 * np.pi)**len(x) * prod_diag)
    return normalization * math.exp(exponent) #P(x|mean,diag_cov)


@njit
def compute_log_likelihood(observations, means, diag_covarinaces, weights): #∑x log(∑k((​wi*​N(x|Ci))

    n_obs = observations.shape[0]
    n_clusters = len(means)
    log_likelihood = 0.0

    for i in range(n_obs): #compute #∑x log(∑k((​wi*​N(x|Ci)) sum of liklihood of all observations 
        x = observations[i]
        log_components = np.empty(n_clusters)
        for k in range(n_clusters):
            mean = means[k] # extract mean of cluster k 
            diag_cov = diag_covarinaces[k]  # extract diagonal of cluster k 
            weight = weights[k]#exctract weight of cluster k 

            # ---- STABILIZATION FIX ----
            if weight < 1e-12:
                weight = 1e-12
            # --------------------------------

            p = probability_density(x, mean, diag_cov) # P(xi|mean_k,diag_cov_k)
            log_components[k] = math.log(weight) + math.log(max(p, 1e-300)) 

        # Log-Sum-Exp
        m = np.max(log_components)
        sum_exp = 0.0
        for v in log_components:
            sum_exp += math.exp(v - m)
        log_likelihood += m + math.log(sum_exp)

    return log_likelihood


@njit(parallel=True)
def maximize_paramters(observations, assignment_scores_k):
    n_observations, n_clusters = assignment_scores_k.shape
    n_features = observations.shape[1]

    means = np.zeros((n_clusters, n_features))
    diag_covariances = np.zeros((n_clusters, n_features)) # nur Diagonalen speichern
    weights = np.zeros(n_clusters)

    for i in prange(n_clusters):
        sum_of_scores = 0.0
        for j in range(n_observations):
            sum_of_scores += assignment_scores_k[j, i]#compute sum of scores over alll observations for an Distribution Ci

        # ---- STABILIZATION FIX ----
        if sum_of_scores < 1e-10:
            sum_of_scores = 1e-10
        # --------------------------------

        # Mean
        for j in range(n_observations):
            means[i] += assignment_scores_k[j, i] * observations[j]
        means[i] /= sum_of_scores #nw mean for feautre i 

        # Covariance (diagonal only)
        for j in range(n_observations):
            diff = observations[j] - means[i]
            diag_covariances[i] += assignment_scores_k[j, i] * (diff**2)
        diag_covariances[i] /= sum_of_scores # new diag_cov 

        # ---- STABILIZATION FIX ----
        # prevent collapsing variances
        for f in range(n_features):
            if diag_covariances[i, f] < 1e-6:
                diag_covariances[i, f] = 1e-6
        # --------------------------------

        # Weight
        weights[i] = sum_of_scores / n_observations #adjsuted weights for  

        # ---- STABILIZATION FIX ----
        if weights[i] < 1e-12:
            weights[i] = 1e-12
        # --------------------------------

    return means, diag_covariances, weights


@njit(parallel=True)
def assignment_score(observations, means, diag, weights): #calculte expectected probability  of xi under all Gaussian distributions 
       
        n_observations = len(observations)
        n_clusters = means.shape[0]

        observations_x_cluster = np.zeros((n_observations,n_clusters))#matrix of P(Ci|xi) 

        for i in range(n_observations):#calculate Probability of observation of xi under all Distributions ci
            x = observations[i]

            # Numerator: Wahrscheinlichkeit für jedes Cluster
            for j in range(n_clusters):
                mean = means[j]
                diag_cov = diag[j]  # extract diagonal
                weight = weights[j]
                
                probability_xi = probability_density(
                    x,
                    mean,  # mean
                    diag_cov   # covariance
                )#P(xi|Ci)
                observations_x_cluster[i, j] = (weight * probability_xi)

            # ------------------ Normalisierung ------------------
            total_prob = 0.0
            for j in range(n_clusters):
                total_prob += observations_x_cluster[i, j]
            
            # ---- STABILIZATION FIX ----
            if total_prob < 1e-300:
                total_prob = 1e-300
            # --------------------------------

            for j in range(n_clusters):
                observations_x_cluster[i, j] /= total_prob  #bayesian rule in order to get  #P(Ci|xi)

        return observations_x_cluster #P(Ci|xi) for alle xi



class Gaussian_Mixture_Model:
   

    @staticmethod
    def GMM(observations, num_of_clusters=5):

        n_observations, n_features = observations.shape 

        means = np.zeros((num_of_clusters, n_features)) #matrix num_of_cluster(rows) , num of feauture values(coulmns)
        diag_covariances = np.zeros((num_of_clusters, n_features))
        weights = np.zeros(num_of_clusters)

        obs_variance = np.var(observations, axis=0)

        # ---- STABILIZATION FIX ----
        obs_variance = np.maximum(obs_variance, 1e-6)
        # --------------------------------

        for k in range(num_of_clusters): 
            random_index = np.random.randint(0, n_observations)
            means[k] = observations[random_index] #builds mean array with k clusters eery mean[k](length is num of feautres)
            diag_covariances[k] = obs_variance  # builds diagonal covariance for every cluster
            weights[k] = 1.0 / num_of_clusters # build weight
        
        log_likelihood = compute_log_likelihood(observations, means, diag_covariances, weights) #compute loglikelihood of GMM over all observations
        log_likelihood_evolution = [log_likelihood]

        # ---------------- EM Iterationen ----------------
        counter = 0
        while True:
            
            counter += 1

            # E-Step: Zugehörigkeitswahrscheinlichkeiten
            expectations = assignment_score(observations, means, diag_covariances, weights)

            # M-Step: Parameter maximieren
            means , diag_covariances , weights  = maximize_paramters(observations, expectations)
            
            # Log-Likelihood berechnen
            previous_log_likelihood = log_likelihood
            log_likelihood = compute_log_likelihood(observations, means, diag_covariances, weights)
            log_likelihood_evolution.append(log_likelihood)

            # Abbruchbedingung
            if abs(previous_log_likelihood - log_likelihood) < 1e-4:
                break

            if counter == 100:
                break
                
        # ---------------- Return ----------------
        return {
             "final_centers": means,
             "weights": weights,
             "log_likelihood_evolution": log_likelihood_evolution,
             "assignments": expectations
         }

def main():
    np.random.seed(42)

    # -------------------------------
    # Generate synthetic data
    # -------------------------------
    n_samples = 300
    cluster1 = np.random.normal(loc=[0, 0], scale=1.0, size=(n_samples, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=1.0, size=(n_samples, 2))
    cluster3 = np.random.normal(loc=[0, 5], scale=1.0, size=(n_samples, 2))

    observations = np.vstack([cluster1, cluster2, cluster3])

    # -------------------------------
    # Fit GMM
    # -------------------------------
    gmm_result = Gaussian_Mixture_Model.GMM(observations, num_of_clusters=3)

    print("Final Cluster Centers:")
    print(gmm_result["final_centers"])
    print("\nCluster Weights:")
    print(gmm_result["weights"])
    print("\nLog-Likelihood Evolution:")
    print(gmm_result["log_likelihood_evolution"])

    # -------------------------------
    # Plot results
    # -------------------------------
    assignments = gmm_result["assignments"]
    cluster_ids = np.argmax(assignments, axis=1)

    plt.figure(figsize=(8, 6))
    for k in range(3):
        plt.scatter(observations[cluster_ids == k, 0],
                    observations[cluster_ids == k, 1],
                    label=f"Cluster {k+1}")
        plt.scatter(gmm_result["final_centers"][k, 0],
                    gmm_result["final_centers"][k, 1],
                    marker='X', s=200, c='black')
    plt.title("GMM Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()