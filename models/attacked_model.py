import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import settings


class AttackedModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layer_dim=500, activation=nn.ReLU()):
        super(AttackedModel, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_layer_dim, bias=True, device=settings.device),
            nn.Linear(hidden_layer_dim, output_dim, bias=False, device=settings.device)
        ])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def plot_decision_boundary(self, dataloader, resolution=100):
        # Set model to evaluation mode
        self.eval()
        self.to(settings.device)

        # Collect all data from the DataLoader
        data, targets = [], []
        for inputs, labels in dataloader:
            data.append(inputs.cpu())
            targets.append(labels.squeeze().cpu())

        data = torch.cat(data).numpy()
        targets = torch.cat(targets).numpy()

        # Create a grid of points to plot the decision boundary
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Convert grid to tensor and move to device
        grid_tensor = torch.tensor(grid, dtype=torch.float32).to(settings.device)

        # Make predictions on the grid
        with torch.no_grad():
            preds = self.forward(grid_tensor).cpu().numpy()
            preds = preds.squeeze()  # Assuming model output is logits or probabilities
            preds = (preds > 0).astype(np.float32)

        # Reshape predictions back to grid shape
        zz = preds.reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, zz, alpha=0.3)

        # Plot the dataset
        plt.scatter(data[:, 0], data[:, 1], c=targets, edgecolor='k', s=20)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Decision Boundary and Margin')
        plt.show()

