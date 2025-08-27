import torch
import numpy as np

def batched_quadratic_form(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix * vector

    A: (N, N) --> (B, N, N)
    x: (B, N) --> (B, N, 1)
    x_T (B, N) --> (B, 1, N)

    returns B x 1 x 1 - each element is x[i].T @ A[i] @ x[i]
    """

    A = A.unsqueeze(0).expand(x.shape[0], -1, -1)   # expand doesn't allocate new memory
    x_T = x.unsqueeze(1).expand(-1, 1, -1)
    x = x.unsqueeze(-1).expand(-1, -1, 1)
    AX = torch.bmm(A, x)
    x_TAX = torch.bmm(x_T, AX)
    return 0.5 * x_TAX.squeeze(-1).squeeze(-1)



if __name__ == "__main__":
    A = torch.eye(5)
    arr = []
    for _ in range(3):
        x = 2.0 * torch.ones(5)
        arr.append(x)
    x = torch.stack(arr)
    print(batched_quadratic_form(A, x))
