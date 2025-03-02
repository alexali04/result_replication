import numpy as np
from EM.utils import fit_gmm, make_cov_matrix, MVG, compute_neg_log_likelihood
import matplotlib.pyplot as plt
import time
from sklearn.mixture import GaussianMixture

np.random.seed(20)

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

X = np.concatenate([x_1, x_2, x_3], axis=0)

print("Starting!")

start = time.time()
Gamma, pi_ks, means, covs = fit_gmm(X, 3, 20, NLL=False)
end = time.time()
print(f"My implementation: Time taken: {end - start} seconds")

Z = np.argmax(Gamma, axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], c=Z, cmap="viridis")
plt.show()

print(f"My implementation NLL: {compute_neg_log_likelihood(X, pi_ks, means, covs):.0f}")
print(f"Means: {means}")
print(f"Covs: {covs}")
print(f"Pi_ks: {pi_ks}")


n_components = 3
model = GaussianMixture(n_components=n_components, random_state=20)
start = time.time()
model.fit(X)
end = time.time()
print(f"SKLearn Time taken: {end - start} seconds")

preds = model.predict(X)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], c=preds, cmap="viridis")
plt.show()

print(f"Means: {model.means_}")
print(f"Covs: {model.covariances_}")
print(f"Pi_ks: {model.weights_}")
print(f"SKLearn NLL: {-1 * model.score(X):.0f}")

print(f"My implementation NLL: {compute_neg_log_likelihood(X, model.weights_, model.means_, model.covariances_):.0f}")






