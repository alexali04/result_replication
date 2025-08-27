"""
We want to understand the effect of preconditioning on loss surfaces in R2

We're using the quadratic form x^T A x

First, we generate circles in the original "x-space"

For ill-conditioned matrices, the level sets are ellipses
- this is because the eigenvalues are disparate so the level sets are long in one direction and short in another

A = Q D Q^T

Q is the eigenvector matrix, rotating basis to align with eigenvectors
D^{1/2} stretches the basis to sqrt(eigenvalues)


We begin with a circle in "eigenspace"
- (z1, z2) = sqrt(c / di) (cos(theta), sin(theta)) where di is the ith eigenvalue of A, c is the level set value
- then, x = Q z. This takes eigen-space circle to x-space ellipse
- then, A^{1/2} x = Q D^{1/2} Q^T (Q z) = Q D^{1/2} z
- this takes x-space ellipse to eigenspace circle but scaled slightly differently

the only reason we do this back and forth is because we want to generate level sets in x-space, normally we just go x-space to z-space
but here we go z-space to x-space to z-space

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A_large = np.array([[50, 6], [5,  2]], dtype=np.float32)  # ill-conditioned matrix - 36.615547
A_small = np.array([[2, 1.0], [1.0,  2]], dtype=np.float32)  # ill-conditioned matrix - 4.2655644
print(np.linalg.cond(A_large))
print(np.linalg.cond(A_small))
print(np.linalg.det(A_small))


def quadratic_form(A, x):
    return 0.5 * x.T @ A @ x 

x_np = np.linspace(-3, 3, 100)
y_np = np.linspace(-3, 3, 100)
X_np, Y_np = np.meshgrid(x_np, y_np)
Z_np_large = np.zeros_like(X_np)
Z_np_small = np.zeros_like(X_np)
for i in range(X_np.shape[0]):
    for j in range(X_np.shape[1]):
        Z_np_large[i, j] = quadratic_form(A_large, np.array([X_np[i, j], Y_np[i, j]]))
        Z_np_small[i, j] = quadratic_form(A_small, np.array([X_np[i, j], Y_np[i, j]]))



fig = plt.figure()
ax = fig.add_subplot(121)
ax.contour(X_np, Y_np, Z_np_large, 50, cmap="RdBu")
ax.plot(0, 0, 0, 'g*', label="global minimum")
ax.set_title(f"Quadratic Form, Condition Number: {np.linalg.cond(A_large):.2f}")
ax.legend()

ax = fig.add_subplot(122)
ax.contour(X_np, Y_np, Z_np_small, 50, cmap="RdBu")
ax.plot(0, 0, 0, 'g*', label="global minimum")
ax.set_title(f"Quadratic Form, Condition Number: {np.linalg.cond(A_small):.2f}")
ax.legend()
plt.show()





exit()

print(np.linalg.cond(A))


# what does the space of the quadratic form look like?

def quadratic_form(x):
    return 0.5 * x.T @ A @ x 

def get_ellipse(c):
    """
    generate circles, apply A to get ellipses - ill-conditioned matrix has elliptical level sets in quadratic form
    need to use eigendecomposition to get the ellipses

    A = Q D Q^T

    x^T A x = x^T Q D Q^T x = (Q^T x)^T D (Q^T x)
    Pick z = Q^T x 

    then, x^T A x = z^T D z = z_1^2 d1 + z_2^2 d2

    z_1^2 d1 + z_2^2 d2 = c

    ellipse of form: u^2 / a^2 + v^2 / b^2 = 1 - u = a cos(theta), v = b sin(theta)

    a = sqrt(c / d1)
    b = sqrt(c / d2)

    so we have: z_1(theta) = sqrt(c / d1) * cos(theta), z_2(theta) = sqrt(c / d2) * sin(theta)
    """
    # generate circles

    thetas = np.linspace(0, 2 * np.pi, 100)

    d, Q = np.linalg.eig(A)
    x = np.sqrt(c / d[0]) * np.cos(thetas)
    y = np.sqrt(c / d[1]) * np.sin(thetas)
    X = np.vstack([x, y])

    Q_X = Q @ X

    return Q_X[0], Q_X[1]

    
iters = 10000
x = np.array([2.0, 2.0])
xy_vals = np.zeros((iters, 2))
for i in range(iters):
    loss = quadratic_form(x)
    
    if i % 500 == 0:
        print(f"Iteration {i + 1}, loss: {loss}")
        print(f"x: {x}")
        xy_vals[i] = x

    # gradient descent
    x = x - 1e-9 * A @ x


x = np.array([0, 0])
loss = quadratic_form(x)
print(f"Loss: {loss}")

x_np = np.linspace(0, 2, 100)
y_np = np.linspace(-0.5, 3, 100)
X_np, Y_np = np.meshgrid(x_np, y_np)
Z_np = np.zeros_like(X_np)
for i in range(X_np.shape[0]):
    for j in range(X_np.shape[1]):
        Z_np[i, j] = quadratic_form(np.array([X_np[i, j], Y_np[i, j]]))

plt.contour(X_np, Y_np, Z_np, 50)
plt.plot(xy_vals[5:, 0], xy_vals[5:, 1], 'r-o')
plt.plot(0, 0, 'g*', label="global minimum")
plt.legend()
plt.show()


# level_sets = [x * 30 for x in range(1, 30)]
# xs = np.zeros((len(level_sets), 100))       
# ys = np.zeros((len(level_sets), 100))

# for i in range(len(level_sets)):
#     xs[i], ys[i] = get_ellipse(level_sets[i])

# zs = np.zeros_like(xs[0])
# for i in range(xs[0].shape[0]):
#     zs[i] = quadratic_form(np.array([xs[0, i], ys[0, i]]))




# fig = plt.figure()
# # ax3d = fig.add_subplot(121, projection="3d")
# ax2d = fig.add_subplot(111)
# for level in range(len(level_sets)):

#     # z_ellipse = np.zeros_like(xs[level])
#     # for i in range(xs[level].shape[0]):
#     #     # for num_points per ellipse
#     #     z_ellipse[i] = quadratic_form(np.array([xs[level, i], ys[level, i]]))
    
#     # ax3d.plot(xs[level], ys[level], z_ellipse, label=f"c={level_sets[level]}")
#     ax2d.plot(xs[level], ys[level], label=f"c={level_sets[level]}")
#     ax2d.plot(xy_vals[:, 0], xy_vals[:, 1], 'r-o')


#     # for i in range(len(xy_vals)):
#     #     ax3d.plot(xy_vals[i][0], xy_vals[i][1], quadratic_form(np.array(xy_vals[i])), 'r-o')


# # ax3d.set_xlabel("x")
# # ax3d.set_ylabel("y")
# # ax3d.set_zlabel("f(x,y)")
# # ax3d.set_title("3D curves (level sets)")

# ax2d.set_xlabel("x")
# ax2d.set_ylabel("y")
# ax2d.set_title("2D curves (gradient descent)")

# plt.show()


