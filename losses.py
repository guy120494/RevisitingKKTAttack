import torch
from torch import nn
from torch.optim import SGD

import settings
from models.attacked_model import AttackedModel


class KKTLoss(nn.Module):
    def __init__(self, model: AttackedModel):
        super(KKTLoss, self).__init__()
        self.model = model

    def compute_loss_of_w(self, x, y, lambdas):
        """
        Compute the expression w_j - v_j * Σ (lambda_i * y_i * x_i * sigma_prime(w_j^T * x_i + b_j))

        Parameters:
            w (torch.Tensor): Tensor of shape (j, d) where j is the number of outputs and d is the feature dimension.
            v (torch.Tensor): Tensor of shape (j,) for each output dimension.
            x (torch.Tensor): Tensor of shape (n, d) where n is the number of samples.
            lambdas (torch.Tensor): Tensor of shape (n,) containing the lambda coefficients.
            y (torch.Tensor): Tensor of shape (n,) containing the y values.
            b (torch.Tensor): Tensor of shape (j,) containing the biases.

        Returns:
            torch.Tensor: The result of the computation.
        """
        w = self.model.get_parameter("layers.0.weight")
        b = self.model.get_parameter("layers.0.bias")
        v = self.model.get_parameter("layers.1.weight")

        # Compute w^T * x + b
        wx_plus_b = torch.matmul(w, x.T) + b[:, None]  # shape (j, n)

        # Derivative of ReLU, which is the Heaviside step function
        # sigma_prime = torch.heaviside(wx_plus_b, torch.tensor(0.0))
        sigma_prime = torch.sigmoid(100 * wx_plus_b)

        # Calculate lambda_i * y_i * x_i * sigma_prime(w_j^T * x_i + b_j)
        # We multiply x (n, d) with sigma_prime (j, n) transposed to align dimensions
        # y and lambda need to be broadcasted correctly
        modified_x = x * lambdas[:, None] * y[:, None]  # element-wise multiplication and broadcasting lambdas and y along d

        # Sum over all samples, resulting in shape (n, d), then we need to sum this for each j
        weighted_sum = torch.einsum('nd,jn->jd', modified_x, sigma_prime)

        # Compute v_j * summed term
        term = v.T * weighted_sum  # broadcasting v over d

        # Subtract from w
        result = (w - term).pow(2).sum()

        return result

    def forward(self, x, targets, lambdas, only_w=True):
        predictions = self.model(x).squeeze()
        assert lambdas.dim() == 1
        assert lambdas.shape == targets.shape
        assert predictions.dim() == 1
        assert targets.dim() == 1

        if 0 in targets:
            targets = 2 * targets - 1

        if only_w:
            inputs = [self.model.get_parameter("layers.0.weight")]
        else:
            inputs = list(self.model.parameters())
        outputs = predictions * targets * lambdas

        return self.compute_loss_of_w(x, targets, lambdas)

        # grads = torch.autograd.grad(
        #     outputs=outputs,
        #     inputs=inputs,
        #     grad_outputs=torch.ones_like(outputs, requires_grad=False, device=settings.device).div(
        #         settings.num_samples),
        #     create_graph=True,
        #     retain_graph=True,
        # )
        #
        # kkt_loss = 0
        # for _, (p, grad) in enumerate(zip(inputs, grads)):
        #     assert p.shape == grad.shape
        #     kkt_loss += (p.detach().data - grad).pow(2).sum()
        #
        # return kkt_loss


class ProjectedGradientOptimizer(SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False, A=None, b=None):
        """
        A Projected Gradient Optimizer that inherits from SGD and projects onto a polyhedron.

        Args:
            params (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
            lr (float): Learning rate.
            momentum (float, optional): Momentum factor (default: 0).
            dampening (float, optional): Dampening for momentum (default: 0).
            weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
            nesterov (bool, optional): Enables Nesterov momentum (default: False).
            A (torch.Tensor, optional): Constraint matrix defining the polyhedron (A @ x <= b).
            b (torch.Tensor, optional): Constraint vector defining the polyhedron (A @ x <= b).
        """
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.A = A
        self.b = b
        if self.A is not None and self.b is not None:
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        param.copy_(self.initialize_interior(param.shape))

    @torch.no_grad()
    def initialize_interior(self, shape):
        """
        Initializes model parameters strictly inside the polyhedron defined by A @ x <= b.

        Args:
            shape (tuple): The desired shape of the parameter tensor.

        Returns:
            torch.Tensor: A tensor with values strictly inside the polyhedron.
        """
        num_variables = self.A.size(1)  # Number of variables (columns of A)

        for _ in range(1000):  # Try up to 1000 attempts to find a valid interior point
            # Generate a random point in the same dimension
            random_vector = torch.randn(num_variables).to(settings.device)
            x = random_vector / random_vector.norm()  # Normalize to unit sphere

            # Scale down the random vector to ensure it's strictly inside the polyhedron
            scale = float('inf')
            for i in range(self.A.size(0)):  # Iterate over constraints
                constraint = self.A[i]
                margin = self.b[i] - constraint @ x
                if margin > 0:
                    scale = min(scale, margin / (constraint @ random_vector).abs().item())

            if scale < float('inf'):
                scale *= 0.9  # Reduce the scale slightly to ensure it's strictly inside
                x = x * scale

            # Check if the result satisfies A @ x < b
            if torch.all(self.A @ x < self.b):
                break
        else:
            raise ValueError("Failed to initialize within the polyhedron interior.")

        # Reshape the parameter to the desired shape
        return x.view(shape)

    @torch.no_grad()
    def project_to_polyhedron(self, param):
        """
        Projects the parameter onto the polyhedron defined by A @ x <= b.

        Args:
            param (torch.Tensor): The parameter tensor to project.

        Returns:
            torch.Tensor: The projected tensor.
        """
        # Flatten parameter tensor for projection
        original_shape = param.shape
        param_flat = param.view(-1)

        # Solve the projection problem using gradient descent
        # Minimize ||x - param||^2 subject to A @ x <= b
        x = param_flat.clone()

        for _ in range(500):  # Max iterations
            violation = self.A @ x - self.b
            if torch.all(violation <= 1e-6):  # No violations
                break

            # Find violated constraints
            mask = violation > 0
            gradient = self.A[mask].sum(dim=0)  # Sum of violated constraint gradients
            step_size = 0.01 / (torch.norm(gradient) + 1e-6)  # Small step size
            x -= step_size * gradient  # Gradient descent step

            # Project back to the original space (ensure numerical stability)
            x = torch.clamp(x, min=param_flat.min(), max=param_flat.max())

        # Reshape back to the original parameter shape
        return x.view(original_shape)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step and applies the projection step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        # Perform the standard SGD step
        loss = super().step(closure)

        # Apply the projection step to all parameters
        if self.A is not None and self.b is not None:
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.copy_(self.project_to_polyhedron(param))

        return loss
