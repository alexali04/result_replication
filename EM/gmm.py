# Gaussian Mixture Model

import numpy as np
from EM.gaussian_utils import MVG, compute_mvg_density
import matplotlib.pyplot as plt

np.random.seed(42)

# Create data

cov_1 = np.identity(n=2)
mean_1 = np.array([0, 0])
mvg1 = MVG(cov_1, mean_1)

cov_2 = np.identity(n=2)
mean_2 = np.array([5, 5])
mvg2 = MVG(cov_2, mean_2)

cov_3 = np.identity(n=2)
mean_3 = np.array([7, -3])
mvg3 = MVG(cov_3, mean_3)

x_1 = mvg1.sample(100)
x_2 = mvg2.sample(80)
x_3 = mvg3.sample(60)

# visualize clusters - cleanly separable
plt.scatter(x_1[:, 0], x_1[:, 1], color="red", label="MVG1")
plt.scatter(x_2[:, 0], x_2[:, 1], color="blue", label="MVG2")
plt.scatter(x_3[:, 0], x_3[:, 1], color="green", label="MVG3")
plt.show()

# run EM - 3 clusters - randomly initialize means, cov, mixing
X = np.concatenate([x_1, x_2, x_3])

k = 3
n = X.shape[0]

covs = np.random.randn(k, 2, 2)
means = np.random.randn(k, 2)
pi_ks = np.random.rand(k)

EM_steps = 100


print(X.shape)

# each E-step: pi_k * N(x_n | mu_k, sigma_k) / sum_k (pi_k * N(x_n | mu_k, sigma_k))
# each M-step: pi_k = 1/n sum_n gamma_nk, mu_k = sum_n gamma_nk x_n / sum_n gamma_nk, sigma_k = sum_n gamma_nk (x_n - mu_k)(x_n - mu_k).T / sum_n gamma_nk

def E_step(X, pi_ks, means, covs):
    Gamma = np.zeros((X.shape[0], k))
    X_probs = np.zeros((X.shape[0]))

    for i in range(X.shape[0]):
        X_probs[i] = np.sum(pi_ks[j] * compute_mvg_density(mu=means[j], Sigma=covs[j], x=X[i], batch=False) for j in range(k))
    
    for i in range(X.shape[0]):
        for j in range(k):
            Gamma[i, j] = pi_ks[j] * compute_mvg_density(mu=means[j], Sigma=covs[j], x=X[i], batch=False) / X_probs[i]
    
    return Gamma

def M_step(X, pi_ks, means, covs, Gamma):
    N = X.shape[0]

    for j in range(k):
        N_k = np.sum(Gamma[:, j]) # N_k = sum_n gamma_nk
        pi_ks[j] = N_k / N        # pi_k = N_k / N

        # mu_k = (sum_n gamma_nk * x_n) / N_k
        means[j] = np.sum(Gamma[n, j] * X[n] for n in range(N)) / N_k 

        # sigma_k = (sum_n gamma_nk * (x_n - mu_k)(x_n - mu_k).T) / N_k
        covs[j] = np.sum(Gamma[n, j] * np.outer(X[n] - means[j], X[n] - means[j]) for n in range(N)) / N_k

for _ in range(100):
    Gamma = E_step(X, pi_ks, means, covs)

    M_step(X, pi_ks, means, covs, Gamma)

    # classification for visualization
    Z = np.argmax(Gamma, axis=1)

    plt.scatter(X[:, 0], X[:, 1], c=Z, cmap="viridis")
    plt.show()




            









