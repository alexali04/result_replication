import numpy as np
import matplotlib.pyplot as plt

import os
import imageio
from typing import Union


class MVG():
    def __init__(self, cov, mean):
        self.cov = cov
        self.mean = mean
    
    def compute_density(self, x, batch=True):
        return compute_mvg_density(mu=self.mean, Sigma=self.cov, x=x, batch=batch)
    
    def sample(self, size):
        rng = np.random.default_rng()
        x_1 = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=size)
        return x_1

    def plot_density(self, x):
        density_plot(mu=self.mean, Sigma=self.cov, x=x)
    
    def get_points_and_densities(self, x):
        """get points"""
        densities = compute_mvg_density(mu=self.mean, Sigma=self.cov, x=x)

        densities = np.expand_dims(densities, axis=1)

        points = np.concat((x, densities), axis=1)
        return points

    
def compute_mvg_density(mu, Sigma, x, batch: bool = True):
    """
    p(x; mu, Sigma) = 1 / {(2 pi)^{k / 2} (det(Sigma))^{1/2}} * exp((x - mu)^T Sigma^{-1} (x - mu))

    x: [B, N]
    mu: [N]
    Sigma: [N, N]
    """
    Sigma_inv = np.linalg.inv(Sigma)
    diff = (x - mu) 

    if batch:
        k = x.shape[1]
        Ax = diff @ Sigma_inv
        exp = np.exp(-0.5 * np.sum(diff * Ax, axis=1)) # avoid wasting O(BN^2) operations
        # exp = np.exp(-0.5 * np.diag((diff @ Sigma_inv) @ diff.T)) 
    else:
        k = x.shape[0]
        exp = np.exp(-0.5 * (diff.T @ Sigma_inv @ diff))

    c = np.power(np.array([2 * np.pi]), np.array([k]))
    det = np.linalg.det(Sigma)

    return exp / np.sqrt(c * det)


def density_plot(mu, Sigma, x):
    densities = compute_mvg_density(mu=mu, Sigma=Sigma, x=x)
    max_density = compute_mvg_density(mu=mu, Sigma=Sigma, x=mu, batch=False)

    densities = np.expand_dims(densities, axis=1)

    points = np.concat((x, densities), axis=1)

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker="o", cmap="inferno", c=points[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')

    ax.set_xlim(min(x[:, 0]) - 0.5, max(x[:, 0]) + 0.5)
    ax.set_ylim(min(x[:, 1]) - 0.5, max(x[:, 1]) + 0.5)
    ax.set_zlim(0, max_density)

    plt.show()


def make_cov_matrix(eigenvalues: np.ndarray, seed: int = 20):
    """
    all eigenvalues are positive for positive definite matrix
    """
    np.random.seed(seed)

    D = np.diag(eigenvalues)
    A = np.random.randn(len(eigenvalues), len(eigenvalues))
    Q, _ = np.linalg.qr(A)

    X = Q @ D @ Q.T
    X = (X + X.T) / 2

    # truncate 6 decimal places
    X *= 1e+6
    X = np.trunc(X)
    X *= 1e-6

    return X


def grid_around_mean(mu, cov, num_points, std_dev=1):
    """
    grid around mean with n_points - 1 standard deviations
    """
    # 1. calculate length of std - scaled Mahalanobis distance
    # 2. create grid around mean bounded by 2 stds
    # 3. return grid

    sigma_inv = np.linalg.inv(cov)

    uv = np.array([0, 1])
    rv = np.array([1, 0])

    a = std_dev * np.sqrt(mu.shape[0]) / np.sqrt(uv.T @ sigma_inv @ uv)
    b = std_dev * np.sqrt(mu.shape[0]) / np.sqrt(rv.T @ sigma_inv @ rv)
    up_vec = mu + a * uv
    right_vec = mu + b * rv
    down_vec = mu - a * uv
    left_vec = mu - b * rv

    len_step_x = np.linalg.norm(right_vec[0] - left_vec[0]) / num_points
    len_step_y = np.linalg.norm(up_vec[1] - down_vec[1]) / num_points

    
    x, y = np.mgrid[left_vec[0]:right_vec[0]:len_step_x, down_vec[1]:up_vec[1]:len_step_y]

    return x, y

# EM utils
def E_step(X, pi_ks, means, covs):
    k = means.shape[0]
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
    k = means.shape[0]

    for j in range(k):
        N_k = np.sum(Gamma[:, j]) # N_k = sum_n gamma_nk
        pi_ks[j] = N_k / N        # pi_k = N_k / N

        # mu_k = (sum_n gamma_nk * x_n) / N_k
        means[j] = np.sum(Gamma[n, j] * X[n] for n in range(N)) / N_k 

        # sigma_k = (sum_n gamma_nk * (x_n - mu_k)(x_n - mu_k).T) / N_k
        covs[j] = np.sum(Gamma[n, j] * np.outer(X[n] - means[j], X[n] - means[j]) for n in range(N)) / N_k


def fit_gmm(X, k, steps, NLL=False):
    """
    Prints neg log likelihood at each step
    """
    covs = np.zeros((k, 2, 2))
    for i in range(k):
        covs[i] = make_cov_matrix(np.array([1, 1]))

    mean_point = np.average(X, axis=0)
    rands = np.random.randn(k, 2)
    means = mean_point + rands * 0.1

    pi_ks = np.random.rand(k)

    for i in range(steps):
        Gamma = E_step(X, pi_ks, means, covs)
        M_step(X, pi_ks, means, covs, Gamma)

        if NLL: 
            neg_log_likelihood = compute_neg_log_likelihood(X, pi_ks, means, covs)
            print(f"Step {i}: {neg_log_likelihood}")
    
    return Gamma, pi_ks, means, covs
    


def compute_neg_log_likelihood(X, pi_ks, means, covs):
    """
    Returns per-sample average neg log likelihood
    """
    N = X.shape[0]
    k = means.shape[0]
    p_x = np.zeros(N)


    for i in range(k):
        densities = compute_mvg_density(mu=means[i], Sigma=covs[i], x=X)
        p_x += pi_ks[i] * densities
    
    return -1 * np.sum(np.log(p_x)) / N

# Plotting utils

def extract_num(
    filename: str
) -> Union[int, float]:
    """
    Extracts number from filename
    """
    base = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, base))
    return int(number) if number.isdigit() else float('inf')

def make_gif(
    folder: str,
    name: str,
    fps: int
) -> None:
    """
    Construct gif 'name' from the images in 'folder'

    Args:
        folder: path to gif folder
        name: name of gif to make
        fps: frames per second
    """

    with imageio.get_writer(f'{name}.gif', mode = 'I', fps = fps, loop=0) as writer:
        for filename in sorted(os.listdir(folder), key=extract_num):
            if filename.endswith('png'):
                image = imageio.imread(folder+"/"+filename)
                writer.append_data(image)
