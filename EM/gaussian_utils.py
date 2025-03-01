import numpy as np
import matplotlib.pyplot as plt

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
        exp = np.exp(-0.5 * np.diag((diff @ Sigma_inv) @ diff.T)) # batch quadratic form
    else:
        k = x.shape[0]
        exp = np.exp(-0.5 * (diff.T @ Sigma_inv @ diff))

    c = np.power(np.array([2 * np.pi]), np.array([k]))
    det = np.linalg.det(Sigma)

    return exp / np.sqrt(c * det)

def get_mvg_samples(mu, Sigma, size):
    rng = np.random.default_rng()
    x_1 = rng.multivariate_normal(mean=mu, cov=Sigma, size=size)
    return x_1

def density_plot(mu, Sigma, x):
    densities = compute_mvg_density(mu=mu, Sigma=Sigma, x=x)
    max_density = compute_mvg_density(mu=mu, Sigma=Sigma, x=mu, batch=False)

    densities = np.expand_dims(densities, axis=1)

    points = np.concat((x, densities), axis=1)

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker="o", color="red")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')

    ax.set_xlim(min(x[:, 0]) - 0.5, max(x[:, 0]) + 0.5)
    ax.set_ylim(min(x[:, 1]) - 0.5, max(x[:, 1]) + 0.5)
    ax.set_zlim(0, max_density)

    plt.show()


