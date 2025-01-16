import torch
from torch import nn
from torch.optim import SGD

import settings
from models.attacked_model import AttackedModel


class KKTLoss(nn.Module):
    def __init__(self, model: AttackedModel):
        super(KKTLoss, self).__init__()
        self.model = model

    def forward(self, predictions, targets, lambdas, only_w=True):
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

        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs, requires_grad=False, device=settings.device).div(settings.num_samples),
            create_graph=True,
            retain_graph=True,
        )

        kkt_loss = 0
        for _, (p, grad) in enumerate(zip(inputs, grads)):
            assert p.shape == grad.shape
            kkt_loss += (p.detach().data - grad).pow(2).sum()

        return kkt_loss


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

        for _ in range(100):  # Max iterations
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
