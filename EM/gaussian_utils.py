import numpy as np
import matplotlib.pyplot as plt


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
        exp = np.exp(-0.5 * np.diag((diff @ Sigma_inv) @ diff.T)) # batch quadratic form
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


