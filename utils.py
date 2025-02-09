import numpy as np
import torch

import settings
from models.attacked_model import AttackedModel


def get_array_of_constraints(model: AttackedModel, training_data_points):
    """
    Return the array of sigma' of the constraints
    :param model: the attacked model
    :param training_data_points: the data used to train the attacked model
    :return: a matrix of constraints. Dimension are (number of neurons) x (number of training examples)
    """
    list_of_w = []
    list_of_b = []
    for name, param in model.named_parameters():
        if name == "layers.0.weight":
            list_of_w = torch.moveaxis(param.clone().detach(), 0, 0)
        if name == "layers.0.bias":
            list_of_b = torch.moveaxis(param.clone().detach(), 0, 0)

    polyhedron_halfspaces = torch.matmul(list_of_w, torch.transpose(training_data_points, 0, 1)) + list_of_b[:, None]
    polyhedron_halfspaces = (polyhedron_halfspaces > 0).type(torch.float)

    return polyhedron_halfspaces


def check_constraints(point, A, b, inequality):
    """
    Check if a point satisfies a set of hyperplane constraints.

    Args:
        point (torch.Tensor): The point to check, shape (d,).
        A (torch.Tensor): Matrix of hyperplane normals, shape (n, d), where n is the number of constraints.
        b (torch.Tensor): Vector of biases, shape (n,).
        inequality (int): Type of inequality, 0 for <= or 1 for >=.

    Returns:
        torch.Tensor: A boolean tensor of shape (n,), where each entry is True if the corresponding constraint is satisfied.
    """
    # Compute the dot product for each hyperplane
    dot_products = torch.matmul(A, point)

    if inequality == 0:
        return dot_products + b <= 0
    elif inequality == 1:
        return dot_products + b >= 0
    else:
        raise ValueError("Invalid inequality type. Use 'leq' for <= or 'geq' for >=.")


def generate_binary_matrices(m, n):
    """
    Generate all possible constraints matrices. Dimension of each matrix is (number of neurons) x (number of training samples)
    :param m: number of neurons in the attacked model
    :param n: number of training samples used to train the attacked model
    """
    total_elements = m * n

    # Generate all combinations of 0s and 1s for the matrix
    import itertools
    all_combinations = itertools.product([0, 1], repeat=total_elements)

    # Iterate over each combination
    for combination in all_combinations:
        # Reshape the combination into an m x n matrix
        matrix = np.array(combination).reshape(m, n)
        yield matrix


def __create_block_matrix(a, n, d):
    # Initialize an empty matrix of zeros with shape (d, n*d)
    matrix = torch.zeros(d, n * d)

    # Fill the matrix according to the pattern
    for i in range(d):
        # Set the elements in the current row
        matrix[i, i * n: (i + 1) * n] = torch.tensor(a)

    return matrix


def create_matrix_for_convex_optimization(model: AttackedModel, list_of_lambdas, sigma_matrix, list_of_y):
    list_of_w = model.state_dict()["layers.0.weight"].clone().detach()
    list_of_v = model.state_dict()["layers.1.weight"].clone().detach().squeeze()

    final_matrix = []
    for j in range(len(list_of_w)):
        a = [list_of_v[j] * list_of_lambdas[i] * list_of_y[i] * sigma_matrix[j, i] for i in range(len(list_of_lambdas))]
        final_matrix.append(__create_block_matrix(a, settings.num_samples, settings.input_dim))
    return torch.vstack(final_matrix).to(settings.device)


def create_polyhedron_boundaries(model: AttackedModel, sigma_matrix):
    list_of_w = model.state_dict()["layers.0.weight"].clone().detach()
    list_of_b = model.state_dict()["layers.0.bias"].clone().detach()

    final_A = []
    final_b = []
    for i in range(sigma_matrix.shape[0]):
        for j in range(sigma_matrix.shape[1]):
            if sigma_matrix[i, j] == 0:
                a = list_of_w[i].clone()
                b = list_of_b[i]
            else:
                a = -list_of_w[i].clone()
                b = -list_of_b[i]

            c = torch.zeros(1, settings.num_samples * settings.input_dim)
            c[:, j:j + settings.input_dim] = a
            final_A.append(c)
            final_b.append(b)
    final_A = torch.vstack(final_A).to(settings.device)
    final_b = torch.vstack(final_b).to(settings.device)

    return final_A, final_b


def get_minimum_norm_solution(X, y):
    """
    Compute the minimum norm solution of the linear system Xx = y using
    the formula X^T (X X^T)^-1 y, implemented in PyTorch.

    Parameters:
        X (torch.Tensor): The coefficient matrix (m x n).
        y (torch.Tensor): The right-hand side vector or matrix (m or m x k).

    Returns:
        x_min (torch.Tensor): The minimum norm solution (n or n x k).
    """
    # Ensure y is a 2D tensor for consistency
    y = y.unsqueeze(-1) if y.ndim == 1 else y

    # Compute the product X X^T
    X_XT = X @ X.T

    # Compute the inverse of X X^T
    X_XT_inv = torch.inverse(X_XT)

    # Compute the minimum norm solution
    x_min = X.T @ (X_XT_inv @ y)

    # If y was a vector, return x_min as a vector
    return x_min.squeeze(-1) if y.shape[1] == 1 else x_min
