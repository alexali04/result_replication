import numpy as np
import matplotlib.pyplot as plt
from EM.gaussian_utils import compute_mvg_density, get_mvg_samples, density_plot


cov_1 = np.identity(n=2)
mean_1 = np.array([0, 0])

cov_2 = np.identity(n=2)
mean_2 = np.array([5, 5])

cov_3 = np.identity(n=2)
mean_3 = np.array([7, -3])

x_1 = get_mvg_samples(mu=mean_1, Sigma=cov_1, size=100)
x_2 = get_mvg_samples(mu=mean_2, Sigma=cov_2, size=80)
x_3 = get_mvg_samples(mu=mean_3, Sigma=cov_3, size=60)

plt.scatter(x_1[:, 0], x_1[:, 1], color="red", label="MVG1")
plt.scatter(x_2[:, 0], x_2[:, 1], color="blue", label="MVG2")
plt.scatter(x_3[:, 0], x_3[:, 1], color="green", label="MVG3")

ax = plt.gca()

ax.set_aspect("equal", adjustable="box")
plt.legend()
plt.show()