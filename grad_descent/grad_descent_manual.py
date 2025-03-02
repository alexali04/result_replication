import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

def L(theta: torch.nn.Parameter, get_grad=False):
    loss = 0.5 * torch.square(torch.square(theta[0]) - theta[1]) + 0.5 * torch.square(theta[0] - 1)
    if get_grad:
        gradient = torch.autograd.grad(loss, theta, grad_outputs=torch.ones_like(loss), create_graph=True)[0]

        return loss, gradient
    
    return loss

def armijo_goldstein(theta: torch.nn.Parameter, c, eta, grad) -> bool:
    """
    direction d = negative gradient

    eta = learning rate
    """

    theta_not = torch.ones_like(theta)

    theta_not.copy_(theta)

    theta_not -= eta * grad


    # check if theta_not theta are diff
    # theta_not += torch.ones_like(theta_not)
    pred = L(theta_not)
    direction_approx = L(theta) + c * eta * ((- grad.T) @ grad)
    # print(f"L predicted: {pred}")
    # print(f"L direction approx: {direction_approx}")
    # print((pred <= direction_approx).item())
    return (pred <= direction_approx).item()
    


theta_0 = torch.Tensor([[0], [0]])
theta_1 = torch.Tensor([[0], [0]])
theta_0.requires_grad = True
theta_1.requires_grad = True
theta_0_param = torch.nn.Parameter(theta_0)
theta_1_param = torch.nn.Parameter(theta_1)

iters = 200
input_output_values_0 = []
input_output_values_1 = []

theta_0_lr = 0.6
theta_1_lr = 0.1
c = 1e-3
not_passed_zero = True
not_passed_one = True

for i in range(iters):
    loss_0, grad_0 = L(theta_0_param, True)
    loss_1, grad_1 = L(theta_1_param, True)

    # backpropagation
    # get the gradient
    # Interior Functions

    theta_0_param = theta_0_param - theta_0_lr * grad_0
    theta_1_param = theta_1_param - theta_1_lr * grad_1

    # if (i + 1) % 10 == 0:
    #     print(theta_0_lr)

    # if not_passed_zero and armijo_goldstein(theta_0_param, c, theta_0_lr, grad_0):
    #     not_passed_zero = False
    #     print("Theta 0 satisfied")
    # else:
    #     theta_0_lr -= c
    
    # if not_passed_one and armijo_goldstein(theta_1_param, c, theta_1_lr, grad_1):
    #     not_passed_one = False
    #     print("Theta 1 passed")
    # else:
    #     # if fail the test and haven't passed it before
    #     theta_1_lr -= c

    input_output_values_0.append((theta_0_param[0].item(), theta_0_param[1].item(), loss_0.item()))
    input_output_values_1.append((theta_1_param[0].item(), theta_1_param[1].item(), loss_1.item()))
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

