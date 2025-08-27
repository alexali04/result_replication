import numpy as np
from EM.utils import fit_gmm, make_cov_matrix, MVG
import matplotlib.pyplot as plt
cov_1 = make_cov_matrix(np.array([1, 1]))
mean_1 = np.array([0, 0])
mvg1 = MVG(cov_1, mean_1)

cov_2 = make_cov_matrix(np.array([1, 1]))
mean_2 = np.array([5, 5])
mvg2 = MVG(cov_2, mean_2)

x_1 = mvg1.sample(100)
x_2 = mvg2.sample(80)

X = np.concatenate([x_1, x_2])
plt.scatter(X[:, 0], X[:, 1])
plt.title("Data")
plt.show()


Gamma, pi_ks, means, covs = fit_gmm(X, 2, 50, NLL=False)

colors = ["mediumseagreen", "mediumorchid"]
Z = np.argmax(Gamma, axis=1)

color_map = [colors[label] for label in Z]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], c=color_map)
plt.title("EM")
plt.show()





