import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[20, 5], [5,  2]], dtype=np.float32)  # ill-conditioned matrix

# what does the space of the quadratic form look like?

def quadratic_form(x):
    return x.T @ A @ x 

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

    



level_sets = [x * 3 for x in range(1, 15)]
xs = np.zeros((len(level_sets), 100))       
ys = np.zeros((len(level_sets), 100))

for i in range(len(level_sets)):
    xs[i], ys[i] = get_ellipse(level_sets[i])

zs = np.zeros_like(xs[0])
for i in range(xs[0].shape[0]):
    zs[i] = quadratic_form(np.array([xs[0, i], ys[0, i]]))




fig = plt.figure()
ax3d = fig.add_subplot(111, projection="3d")

for level in range(len(level_sets)):

    z_ellipse = np.zeros_like(xs[level])
    for i in range(xs[level].shape[0]):
        # for num_points per ellipse
        z_ellipse[i] = quadratic_form(np.array([xs[level, i], ys[level, i]]))
    
    ax3d.plot(xs[level], ys[level], z_ellipse, label=f"c={level_sets[level]}")


ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("f(x,y)")
ax3d.set_title("3D curves (level sets)")

plt.show()


