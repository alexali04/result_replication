import torch
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

def L(theta: torch.nn.Parameter):
    """
    analytically, minimized when theta_0 = 1, theta_1 = 1

    L = 0.5 * (theta_0^2 - theta_1)^2 + 0.5 * (theta_0 - 1)^2
    """
    return 0.5 * torch.square(torch.square(theta[0]) - theta[1]) + 0.5 * torch.square(theta[0] - 1)
    
theta_0 = torch.Tensor([[0], [0]])
theta_1 = torch.Tensor([[0], [0]])
theta_0.requires_grad = True
theta_1.requires_grad = True
theta_0_param = torch.nn.Parameter(theta_0)
theta_1_param = torch.nn.Parameter(theta_1)

iters = 200
optimizer_0 = SGD([theta_0_param], lr=0.6)
optimizer_1 = SGD([theta_1_param], lr=0.1)
input_output_values_0 = []
input_output_values_1 = []
for i in range(iters):
    optimizer_0.zero_grad()
    optimizer_1.zero_grad()
    loss_0 = L(theta_0_param)
    loss_1 = L(theta_1_param)
    loss_0.backward()
    loss_1.backward()
    optimizer_0.step()
    optimizer_1.step()
    print(f"0: Iter {i}: loss={loss_0.item():.4f}, θ0={theta_0_param[0].item():.4f}, θ1={theta_0_param[1].item():.4f}")
    print(f"1: Iter {i}: loss={loss_1.item():.4f}, θ0={theta_1_param[0].item():.4f}, θ1={theta_1_param[1].item():.4f}")
    input_output_values_0.append((theta_0_param[0].item(), theta_0_param[1].item(), loss_0.item()))
    input_output_values_1.append((theta_1_param[0].item(), theta_1_param[1].item(), loss_1.item()))

x_np = np.linspace(0, 2, 100)
y_np = np.linspace(-0.5, 3, 100)
X_np, Y_np = np.meshgrid(x_np, y_np)
Z_np = L(torch.Tensor([X_np, Y_np]))
x_coords, y_coords, z_coords = zip(*input_output_values_0)
x_coords_1, y_coords_1, z_coords_1 = zip(*input_output_values_1)

plt.contour(X_np, Y_np, Z_np, 50)
plt.plot(1, 1, "g*", label="global minimum")
plt.plot(x_coords, y_coords, "b->", label="SGD, learning rate=0.6")
plt.plot(x_coords_1, y_coords_1, "r->", label="SGD, learning rate=0.1")
plt.legend()
plt.show()



x = np.linspace(-1, 3, 100)
y = np.linspace(-1, 3, 100)
x, y = np.meshgrid(x, y)
z = L(torch.Tensor([x, y]))
z = np.minimum(z, 10)




fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.3)
# ax.contour(X_np, Y_np, Z_np, levels=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 1, 2, 5, 7, 10], zdir='z', offset=-2, cmap='coolwarm')
ax.plot(1, 1, 0, c='green', marker='o', label='global minimum')
ax.plot(x_coords, y_coords, z_coords, "b->", label='SGD, learning rate=0.6')
ax.plot(x_coords_1, y_coords_1, z_coords_1, "r->", label='SGD, learning rate=0.1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(-1, 10)
ax.legend()
plt.show()


