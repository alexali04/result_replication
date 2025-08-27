import numpy as np
import matplotlib.pyplot as plt
from EM.utils import MVG

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

p1 = mvg1.get_points_and_densities(x_1)
p2 = mvg2.get_points_and_densities(x_2)
p3 = mvg3.get_points_and_densities(x_3)


# point plots
plt.scatter(x_1[:, 0], x_1[:, 1], color="red", label="MVG1")
plt.scatter(x_2[:, 0], x_2[:, 1], color="blue", label="MVG2")
plt.scatter(x_3[:, 0], x_3[:, 1], color="green", label="MVG3")

ax = plt.gca()

ax.set_aspect("equal", adjustable="box")
plt.legend()
plt.show()

# density plots
fig = plt.figure()

ax = fig.add_subplot(projection="3d")

ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], cmap="autumn", c=p1[:, 2], label="MVG1")
ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], cmap="winter", c=p2[:, 2], label="MVG2")
ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], cmap="summer", c=p3[:, 2], label="MVG3")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density')

plt.legend()
plt.title("Isotropic Gaussian Density Plot")
plt.show()
