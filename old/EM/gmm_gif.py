import numpy as np
from EM.utils import MVG, compute_mvg_density, make_gif, make_cov_matrix, grid_around_mean, E_step, M_step, compute_neg_log_likelihood
import matplotlib.pyplot as plt
import os
import shutil


np.random.seed(20)
# Create data

cov_1 = make_cov_matrix(np.array([1, 1]))
mean_1 = np.array([0, 0])
mvg1 = MVG(cov_1, mean_1)

cov_2 = make_cov_matrix(np.array([1, 1]))
mean_2 = np.array([5, 5])
mvg2 = MVG(cov_2, mean_2)

cov_3 = make_cov_matrix(np.array([1, 1]))
mean_3 = np.array([7, -2])
mvg3 = MVG(cov_3, mean_3)

true_means = np.concatenate([mean.reshape(1, -1) for mean in [mean_1, mean_2, mean_3]], axis=0)

x_1 = mvg1.sample(100)
x_2 = mvg2.sample(80)
x_3 = mvg3.sample(60)

# visualize clusters - cleanly separable
plt.scatter(x_1[:, 0], x_1[:, 1], color="red", label="MVG1")
plt.scatter(x_2[:, 0], x_2[:, 1], color="blue", label="MVG2")
plt.scatter(x_3[:, 0], x_3[:, 1], color="green", label="MVG3")
plt.show()

X = np.concatenate([x_1, x_2, x_3])
X_min, X_max = np.min(X[:, 0]), np.max(X[:, 0])
Y_min, Y_max = np.min(X[:, 1]), np.max(X[:, 1])

# run EM - 3 clusters - randomly initialize means, cov, mixing
k = 3
n = X.shape[0]

covs = np.zeros((k, 2, 2))
for i in range(k):
    covs[i] = make_cov_matrix(np.array([1, 1]))

mean_point = np.average(X, axis=0)
rands = np.random.randn(k, 2)
means = mean_point + rands * 0.1

pi_ks = np.random.rand(k)

EM_steps = 100



FOLDER = "./EM/images"
if os.path.exists(FOLDER):
    shutil.rmtree(FOLDER)
os.makedirs(FOLDER)

for em in range(0, 22):
    Gamma = E_step(X, pi_ks, means, covs)

    if em > 0:
        M_step(X, pi_ks, means, covs, Gamma) # skipping an M step doesn't matter because the E step is invariant to it

    # classification for visualization
    Z = np.argmax(Gamma, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cluster_colors = ["red", "blue", "green"]
    density_cmaps = ["autumn", "winter", "summer"]
    mean_colors = ["indianred", "mediumblue", "forestgreen"]

    # plot data
    data_colors = [cluster_colors[label] for label in Z]
    ax.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.25)

    # iterating over gaussians
    for i in range(k):
        x, y = grid_around_mean(means[i], covs[i], 10) 
        pos = np.dstack((x, y)).reshape(-1, 2)
        densities = compute_mvg_density(mu=means[i], Sigma=covs[i], x=pos)
        densities = densities.reshape(x.shape)
        cs = ax.contour(x, y, densities, levels=8, cmap=density_cmaps[i], alpha=0.5)

        ax.scatter(true_means[i, 0], true_means[i, 1], c="black", marker="x", s=100)
        ax.scatter(means[i, 0], means[i, 1], c=mean_colors[i], marker="x", s=100, label=f"MVG {i}:{pi_ks[i]:.2f}")

    ax.set_xlim(X_min - 0.5, X_max + 0.5)
    ax.set_ylim(Y_min - 0.5, Y_max + 0.5)
    ax.set_title(f"EM iteration {em}, NLL: {compute_neg_log_likelihood(X, pi_ks, means, covs):.3f}")
    ax.legend(loc="upper left")
    plt.savefig(f"{FOLDER}/{str(em)}.png")
    plt.close()


make_gif(FOLDER, "GMM", 2)


            









